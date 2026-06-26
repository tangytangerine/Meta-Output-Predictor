#!/usr/bin/env python3
"""
Single script for training and testing a transformer on LTI trajectories.

This script provides the core functionality of the Meta-Output-Predictor repository:
- Training a transformer (GPT2-based) on LTI trajectories
- Testing on another set of trajectories with customizable settings
- Support for different training and test settings (noise levels, dimensions, etc.)

Usage:
    # As a standalone script
    python train_test_lti.py --num_train_trajectories 100 --sigma_w 0.1 --test_sigma_w 0.2
    
    # Or import and use programmatically
    from train_test_lti import TrainingConfig, train_and_test
    config = TrainingConfig()
    config.num_train_trajectories = 100
    results = train_and_test(config)
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning import loggers as pl_loggers
from transformers import GPT2Model, GPT2Config
import hashlib
import time
import logging
from typing import Optional, Dict, Any, Tuple, List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


class TrainingConfig:
    """Configuration for training and testing LTI trajectory transformer."""
    
    def __init__(self):
        # Dataset settings for training
        self.num_train_trajectories = 20000
        self.num_val_trajectories = 5000
        self.traj_length = 50  # n_positions
        self.nx = 10  # state dimension
        self.ny = 5   # observation dimension
        self.n_noise = 1
        self.sigma_w = 1e-1  # process noise
        self.sigma_v = 1e-1  # observation noise
        self.dataset_type = "ypred"  # or "noniid", "upperTriA"
        
        # Model settings
        self.n_dims_in = None  # Will be auto-calculated as 2*nx (state[:-1] + inputs)
        self.n_dims_out = None  # Will be auto-calculated as nx (predict states)
        self.n_embd = 256
        self.n_layer = 12
        self.n_head = 8
        self.n_positions = 50
        
        # Training settings
        self.batch_size = 64
        self.num_epochs = 1
        self.learning_rate = 3e-4
        self.weight_decay = 1e-2
        self.gradient_clip_val = 1.0
        self.gradient_clip_algorithm = 'norm'
        
        # Test settings (can be overridden independently)
        self.num_test_trajectories = 100
        self.test_traj_length = 50
        self.test_nx = None  # If None, uses training nx
        self.test_ny = None  # If None, uses training ny
        self.test_sigma_w = None  # If None, uses training sigma_w
        self.test_sigma_v = None  # If None, uses training sigma_v
        self.test_n_noise = None  # If None, uses training n_noise
        
        # Other settings
        self.seed = 42
        self.output_dir = None  # If None, auto-generated
        self.ckpt_path = None  # Checkpoint to resume from
        self.train_data_workers = 4
        self.val_data_workers = 2
        
    def get_test_config(self) -> dict:
        """Get test configuration, falling back to training config where not specified."""
        return {
            'n_positions': self.test_traj_length,
            'nx': self.test_nx if self.test_nx is not None else self.nx,
            'ny': self.test_ny if self.test_ny is not None else self.ny,
            'sigma_w': self.test_sigma_w if self.test_sigma_w is not None else self.sigma_w,
            'sigma_v': self.test_sigma_v if self.test_sigma_v is not None else self.sigma_v,
            'n_noise': self.test_n_noise if self.test_n_noise is not None else self.n_noise,
            'dataset_typ': self.dataset_type,
        }
    
    def calculate_dims(self):
        """Auto-calculate n_dims_in and n_dims_out based on nx, ny, and dataset type."""
        if self.n_dims_in is None:
            # For ypred, noniid, upperTriA: xs = [state[:-1], inputs], ys = state[1:]
            # inputs have dimension nx, state has dimension nx
            self.n_dims_in = self.nx + self.nx  # state[:-1] + inputs
        if self.n_dims_out is None:
            self.n_dims_out = self.nx  # predict states


class FilterSim:
    """LTI system simulation with noise and inputs."""
    
    def __init__(self, nx=3, ny=2, sigma_w=1e-1, sigma_v=1e-1, tri=False, n_noise=1, seed=None):
        self.sigma_w = sigma_w
        self.sigma_v = sigma_v
        self.n_noise = n_noise
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random A matrix
        if tri:
            rng = np.random.default_rng(seed)
            lims = np.array([0.2, 0.9])
            A = np.diag(rng.uniform(lims[0], lims[-1], (nx)))
            A[np.triu_indices(nx, 1)] = rng.uniform(lims[0], lims[-1], (nx**2+nx)//2-nx)
            self.A = A
        else:
            self.A = self._random_nonsymmetric_with_eigs(nx, 0.2, 0.9, seed=seed)
        
        # Generate C matrix ensuring observability
        self.C = np.eye(nx) if nx == ny else self._construct_C(self.A, ny)
    
    @staticmethod
    def _random_nonsymmetric_with_eigs(n, eig_low, eig_high, cond_max=1e3, seed=None, max_tries=10000):
        rng = np.random.default_rng(seed)
        eigs = rng.uniform(eig_low, eig_high, size=n)
        Lambda = np.diag(eigs)
        
        for _ in range(max_tries):
            P = rng.normal(size=(n, n))
            if np.linalg.matrix_rank(P) < n:
                continue
            c = np.linalg.cond(P)
            if c <= cond_max:
                A = P @ Lambda @ np.linalg.inv(P)
                return A
        raise RuntimeError("Could not find a well-conditioned P within max_tries")
    
    @staticmethod
    def _construct_C(A, ny):
        nx = A.shape[0]
        _O = [np.eye(nx)]
        for _ in range(nx-1):
            _O.append(_O[-1] @ A)
        while True:
            C = np.random.rand(ny, nx)
            O = np.concatenate([C @ o for o in _O], axis=0)
            if np.linalg.matrix_rank(O) == nx:
                break
        return C.astype('f')
    
    def simulate(self, traj_len, x0=None):
        """Simulate a trajectory with process noise, observation noise, and inputs."""
        ny, nx = self.C.shape
        n_noise = self.n_noise
        
        if x0 is None:
            x0 = np.random.randn(nx)
        
        # Initial state
        xs = [x0.astype('f')]
        xs_cl = [x0.astype('f')]  # clean (no process noise)
        vs = [(np.random.randn(ny)) * self.sigma_v for _ in range(n_noise)]
        ws = [(np.random.randn(nx)) * self.sigma_w for _ in range(n_noise)]
        ys = [self.C @ xs[0] + sum(vs)]
        ys_cl = [self.C @ xs_cl[0] + sum(vs)]
        
        # Inputs
        us = []
        for _ in range(traj_len):
            # Generate random input (control signal) - small range to prevent explosion
            ulims = np.array([-0.1, 0.1])
            u = np.random.uniform(ulims[0], ulims[1], size=nx)
            us.append(u.astype('f'))
            
            # State update with process noise and input
            x_cl = self.A @ xs_cl[-1] + sum(ws[-n_noise:])
            x = self.A @ xs[-1] + sum(ws[-n_noise:]) + u
            
            xs_cl.append(x_cl)
            xs.append(x)
            ws.append((np.random.randn(nx)) * self.sigma_w)
            
            vs.append((np.random.randn(ny)) * self.sigma_v)
            y_cl = self.C @ xs_cl[-1] + sum(vs[-n_noise:])
            y = self.C @ xs[-1] + sum(vs[-n_noise:])
            ys_cl.append(y_cl)
            ys.append(y)
        
        return (
            np.array(xs).astype("f"), 
            np.array(ys).astype("f"), 
            np.array(us).astype("f"), 
            np.array(xs_cl).astype("f"), 
            np.array(ys_cl).astype("f")
        )


def check_validity(entry: dict) -> bool:
    """Check if trajectory is valid (not exploding)."""
    if entry is None:
        return False
    states = entry.get("states")
    obs = entry.get("obs")
    if states is None or obs is None:
        return False
    return np.max(np.abs(states)) < 50 and np.max(np.abs(obs)) < 50


def generate_lti_sample(dataset_type: str, n_positions: int, nx: int, ny: int, 
                        sigma_w: float = 1e-1, sigma_v: float = 1e-1, 
                        n_noise: int = 1, seed: int = None) -> Tuple[FilterSim, dict]:
    """Generate a single LTI trajectory sample."""
    tri = (dataset_type == "upperTriA")
    
    while True:
        fsim = FilterSim(nx=nx, ny=ny, sigma_w=sigma_w, sigma_v=sigma_v, 
                        tri=tri, n_noise=n_noise, seed=seed)
        states, obs, us, states_cl, obs_cl = fsim.simulate(n_positions)
        entry = {
            "states": states, 
            "inputs": us, 
            "obs": obs, 
            "statesCL": states_cl, 
            "obsCL": obs_cl, 
            "A": fsim.A, 
            "C": fsim.C
        }
        if check_validity(entry):
            return fsim, entry


def generate_dataset(num_samples: int, config: dict, dataset_type: str = "train") -> List[dict]:
    """Generate a dataset of LTI trajectories."""
    samples = []
    
    # Use tqdm if available
    try:
        from tqdm import tqdm
        iterator = tqdm(range(num_samples), desc=f"Generating {dataset_type} samples")
    except ImportError:
        iterator = range(num_samples)
    
    for i in iterator:
        seed = config.get('seed', None)
        if seed is not None:
            # Vary seed for each sample
            actual_seed = seed + i
        else:
            actual_seed = None
            
        _, entry = generate_lti_sample(
            dataset_type=config.get('dataset_typ', 'ypred'),
            n_positions=config['n_positions'],
            nx=config['nx'],
            ny=config['ny'],
            sigma_w=config['sigma_w'],
            sigma_v=config['sigma_v'],
            n_noise=config['n_noise'],
            seed=actual_seed
        )
        # Remove clean trajectories
        entry.pop("statesCL", None)
        entry.pop("obsCL", None)
        samples.append(entry)
    
    return samples


class LTIDataset(Dataset):
    """PyTorch Dataset for LTI trajectories."""
    
    def __init__(self, samples: List[dict], config: TrainingConfig):
        self.samples = samples
        self.config = config
        self.dataset_type = config.dataset_type
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        entry = self.samples[idx % len(self.samples)].copy()
        
        if self.dataset_type in ["ypred", "noniid", "upperTriA"]:
            sta = entry.pop("states")
            obs = entry.pop("obs")
            actions = entry.pop("inputs")
            # Input is [state[:-1], action], output is state[1:]
            entry["xs"] = np.concatenate([sta[:-1], actions], axis=-1)
            entry["ys"] = sta[1:]
        else:
            raise NotImplementedError(f"{self.dataset_type} is not implemented")
        
        # Convert to torch tensors
        torch_entry = {
            k: torch.from_numpy(a).float() if isinstance(a, np.ndarray) else a
            for k, a in entry.items()
        }
        return torch_entry


class LTIDataModule(pl.LightningDataModule):
    """Data module for LTI trajectory training."""
    
    def __init__(self, train_ds: Optional[LTIDataset] = None, val_ds: Optional[LTIDataset] = None,
                 config: TrainingConfig = None, batch_size: int = 64):
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.config = config
        self.batch_size = batch_size
    
    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.config.train_data_workers if self.config else 4,
            pin_memory=True,
            drop_last=False,
        )
    
    def val_dataloader(self):
        if self.val_ds is not None:
            return DataLoader(
                self.val_ds,
                batch_size=self.batch_size * 2,
                shuffle=False,
                num_workers=self.config.val_data_workers if self.config else 2,
                pin_memory=True,
                drop_last=False,
            )
        return None
    
    def test_dataloader(self):
        if self.val_ds is not None:
            return DataLoader(
                self.val_ds,
                batch_size=self.batch_size * 2,
                shuffle=False,
                num_workers=self.config.val_data_workers if self.config else 2,
                pin_memory=True,
                drop_last=False,
            )
        return None


class GPT2Transformer(pl.LightningModule):
    """GPT2-based transformer model for predicting next states from current states and inputs."""
    
    def __init__(self, n_dims_in: int, n_positions: int, 
                 n_dims_out: int = 10, n_embd: int = 256, 
                 n_layer: int = 4, n_head: int = 8):
        super(GPT2Transformer, self).__init__()
        
        self.save_hyperparameters()
        
        self.n_positions = n_positions
        self.n_dims_in = n_dims_in
        self.n_dims_out = n_dims_out
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        
        # Create GPT2 configuration
        gpt2_config = GPT2Config(
            n_positions=2048,  # Set large for flexibility
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"
        
        # Model layers
        self._read_in = nn.Linear(n_dims_in, n_embd)
        self._backbone = GPT2Model(gpt2_config)
        self._read_out = nn.Linear(n_embd, n_dims_out)
    
    def predict_step(self, input_dict: dict) -> Tuple[dict, dict]:
        """Predict next states from input."""
        xs = input_dict["xs"]  # (batch, seq_len, n_dims_in)
        
        # Embed input
        embeds = self._read_in(xs)  # (batch, seq_len, n_embd)
        
        # Pass through transformer
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        
        # Predict output
        prediction = self._read_out(output)  # (batch, seq_len, n_dims_out)
        
        return input_dict, {"preds": prediction}
    
    def forward(self, input_dict: dict, batch_idx: Optional[int] = None, 
                return_intermediate_dict: bool = False):
        input_dict, intermediate_dict = self.predict_step(input_dict, batch_idx)
        
        # Calculate losses
        output_dict = self.calculate_losses_and_metrics(input_dict, intermediate_dict)
        
        # Calculate optimized loss (sum of all loss terms)
        optimized_loss = 0
        for key, loss in output_dict.items():
            if "loss_" in key:
                optimized_loss += loss
        output_dict["optimized_loss"] = optimized_loss
        
        if return_intermediate_dict:
            return intermediate_dict, output_dict
        return output_dict
    
    def calculate_losses_and_metrics(self, input_dict: dict, intermediate_dict: dict) -> dict:
        """Calculate MSE loss between predictions and ground truth."""
        ys = input_dict["ys"]  # Ground truth next states
        preds = intermediate_dict["preds"]
        
        # MSE loss
        res_sq = (preds - ys) ** 2
        output_dict = {}
        output_dict["loss_mse"] = torch.mean(res_sq)
        
        # Per-timestep, per-dimension metrics
        for i in range(ys.shape[1]):
            for j in range(ys.shape[2]):
                output_dict[f"metric_mse_ts{i}_dim_{j}"] = torch.mean(res_sq[:, i, j])
        
        return output_dict
    
    def predict_ar(self, inputs: np.ndarray, fix_window_len: bool = True) -> np.ndarray:
        """
        Autoregressive prediction on a single trajectory.
        
        Args:
            inputs: Input array of shape (traj_len, n_dims_in) or (batch, traj_len, n_dims_in)
            fix_window_len: If True, only use the last n_positions inputs for prediction
            
        Returns:
            Predictions of shape (batch, traj_len+1, n_dims_out) or (traj_len+1, n_dims_out)
        """
        inputs_t = torch.from_numpy(inputs).float().to(self.device)
        one_d = False
        if inputs_t.ndim == 2:
            one_d = True
            inputs_t = inputs_t.unsqueeze(0)
        
        bsize, points, _ = inputs_t.shape
        d_o = self.n_dims_out
        
        # Initialize output with zeros for first timestep
        outs = torch.zeros(bsize, 1, d_o).to(self.device)
        
        with torch.no_grad():
            for i in range(1, points + 1):
                # Take first i inputs
                I = inputs_t[:, :i]
                if fix_window_len and I.shape[1] > self.n_positions:
                    I = I[:, -self.n_positions:]
                
                _, interm = self.predict_step({"xs": I})
                pred = interm["preds"][:, -1:]  # Take last prediction
                outs = torch.cat([outs, pred], dim=1)
        
        outs = outs.detach().cpu().numpy()
        if one_d:
            outs = outs[0]
        return outs
    
    def training_step(self, input_dict, batch_idx):
        intermediate_dict, output_dict = self(
            input_dict, batch_idx=batch_idx, return_intermediate_dict=True)
        self.log_output_dct(output_dict, "train")
        return {"loss": output_dict["optimized_loss"],
                "intermediate_dict": intermediate_dict,
                "output_dict": output_dict}
    
    def validation_step(self, input_dict, batch_idx):
        intermediate_dict, output_dict = self(
            input_dict, batch_idx=batch_idx, return_intermediate_dict=True)
        self.log_output_dct(output_dict, "val")
        return {"loss": output_dict["optimized_loss"],
                "intermediate_dict": intermediate_dict,
                "output_dict": output_dict}
    
    def test_step(self, input_dict, batch_idx):
        output_dict = self(input_dict, batch_idx=batch_idx)
        self.log_output_dct(output_dict, "test")
    
    def log_output_dct(self, output_dict: dict, typ: str):
        """Log all loss and metric values."""
        for k in output_dict:
            if "loss" in k or "metric" in k:
                self.log(typ + "_" + k, output_dict[k], on_step=True, on_epoch=True,
                         prog_bar=True, logger=True)
    
    def configure_optimizers(self):
        """Configure optimizer with weight decay."""
        config = getattr(self, 'config', None)
        lr = config.learning_rate if config else 3e-4
        wd = config.weight_decay if config else 1e-2
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)
        return optimizer


def setup_train(model: GPT2Transformer, config: TrainingConfig) -> str:
    """Set up training output directory."""
    output_dir = config.ckpt_path
    if output_dir is not None and output_dir != '':
        output_dir = "/".join(output_dir.split("/")[:-2])
        logger.info(f'Resuming from the checkpoint: {config.ckpt_path}')
    
    if output_dir is None:
        identifier = (model.__class__.__name__ + '/' +
                    time.strftime('%y%m%d_%H%M%S') + '.' +
                    hashlib.md5(str(config.__dict__).encode('utf-8')).hexdigest()[:6])
        output_dir = os.path.join('.', 'outputs', identifier)
        
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
    
    # Log messages to file
    root_logger = logging.getLogger()
    file_handler = logging.FileHandler(os.path.join(output_dir, 'messages.log'))
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(formatter)
    for handler in root_logger.handlers[1:]:  # all except stdout
        root_logger.removeHandler(handler)
    root_logger.addHandler(file_handler)
    
    # Print model details
    num_params = sum([
        np.prod(p.size())
        for p in filter(lambda p: p.requires_grad, model.parameters())
    ])
    logger.info(f'\nThere are {num_params} trainable parameters.\n')
    
    return output_dir


def get_callbacks_and_loggers(model: GPT2Transformer, output_dir: str, config: TrainingConfig):
    """Get training callbacks and loggers."""
    lr_monitor = pl_callbacks.LearningRateMonitor(logging_interval='epoch')
    tb_logger = pl_loggers.TensorBoardLogger(output_dir)
    loggers = [tb_logger]
    
    checkpoint_callback = pl_callbacks.ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        filename="{step}",
        save_top_k=-1,
        every_n_train_steps=10000,
    )
    
    callbacks = [checkpoint_callback, lr_monitor]
    return callbacks, loggers


def train_model(config: TrainingConfig, model: Optional[GPT2Transformer] = None) -> Tuple[GPT2Transformer, str]:
    """
    Train the transformer model on LTI trajectories.
    
    Args:
        config: Training configuration
        model: Optional pre-initialized model
        
    Returns:
        Tuple of (trained model, output directory)
    """
    # Auto-calculate dimensions if not specified
    config.calculate_dims()
    
    # Set random seed for reproducibility
    if config.seed is not None:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
    
    # Generate training data
    logger.info(f"Generating {config.num_train_trajectories} training trajectories...")
    train_samples = generate_dataset(
        num_samples=config.num_train_trajectories,
        config={
            'n_positions': config.traj_length,
            'nx': config.nx,
            'ny': config.ny,
            'sigma_w': config.sigma_w,
            'sigma_v': config.sigma_v,
            'n_noise': config.n_noise,
            'dataset_typ': config.dataset_type,
            'seed': config.seed,
        },
        dataset_type="train"
    )
    
    # Generate validation data
    logger.info(f"Generating {config.num_val_trajectories} validation trajectories...")
    val_samples = generate_dataset(
        num_samples=config.num_val_trajectories,
        config={
            'n_positions': config.traj_length,
            'nx': config.nx,
            'ny': config.ny,
            'sigma_w': config.sigma_w,
            'sigma_v': config.sigma_v,
            'n_noise': config.n_noise,
            'dataset_typ': config.dataset_type,
            'seed': config.seed + 10000,  # Different seed for validation
        },
        dataset_type="val"
    )
    
    # Create datasets
    train_ds = LTIDataset(train_samples, config)
    val_ds = LTIDataset(val_samples, config)
    
    # Create data module
    datamodule = LTIDataModule(train_ds, val_ds, config, config.batch_size)
    
    # Initialize model if not provided
    if model is None:
        model = GPT2Transformer(
            n_dims_in=config.n_dims_in,
            n_positions=config.n_positions,
            n_dims_out=config.n_dims_out,
            n_embd=config.n_embd,
            n_layer=config.n_layer,
            n_head=config.n_head
        )
    
    # Store config in model for optimizer
    model.config = config
    
    # Set up training
    output_dir = setup_train(model, config)
    logger.info(f"Output directory: {output_dir}")
    callbacks, loggers = get_callbacks_and_loggers(model, output_dir, config)
    
    # Create trainer
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=loggers,
        devices="auto",
        accelerator="auto",
        gradient_clip_algorithm=config.gradient_clip_algorithm,
        gradient_clip_val=config.gradient_clip_val,
        log_every_n_steps=50,
        max_epochs=config.num_epochs
    )
    
    # Train
    logger.info("Starting training...")
    ckpt_path = config.ckpt_path if config.ckpt_path != '' else None
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    
    return model, output_dir


def test_model(model: GPT2Transformer, config: TrainingConfig) -> Dict[str, Any]:
    """
    Test the model on LTI trajectories with potentially different settings.
    
    Args:
        model: Trained model
        config: Training configuration (test settings can differ from training)
        
    Returns:
        Dictionary with test results and samples
    """
    # Get test configuration (may differ from training)
    test_config = config.get_test_config()
    
    logger.info(f"\nTesting with configuration:")
    logger.info(f"  Trajectory length: {test_config['n_positions']}")
    logger.info(f"  State dimension (nx): {test_config['nx']}")
    logger.info(f"  Observation dimension (ny): {test_config['ny']}")
    logger.info(f"  Process noise (sigma_w): {test_config['sigma_w']}")
    logger.info(f"  Observation noise (sigma_v): {test_config['sigma_v']}")
    logger.info(f"  Number of noise sources: {test_config['n_noise']}")
    
    # Generate test data
    logger.info(f"\nGenerating {config.num_test_trajectories} test trajectories...")
    test_samples = generate_dataset(
        num_samples=config.num_test_trajectories,
        config=test_config,
        dataset_type="test"
    )
    
    # Create test dataset and data module
    test_ds = LTIDataset(test_samples, config)
    test_datamodule = LTIDataModule(None, test_ds, config, config.batch_size)
    
    # Create trainer for testing
    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        logger=False,
        enable_checkpointing=False,
    )
    
    # Run test
    logger.info("Running test...")
    test_results = trainer.test(model, datamodule=test_datamodule)
    
    # Also run autoregressive prediction on a few samples
    logger.info("\nRunning autoregressive predictions...")
    ar_results = []
    num_ar_samples = min(5, len(test_samples))
    for i in range(num_ar_samples):
        sample = test_samples[i]
        sta = sample["states"]
        us = sample["inputs"]
        
        # Create input: [state[:-1], action]
        xs = np.concatenate([sta[:-1], us], axis=-1)
        
        # Predict
        preds = model.predict_ar(xs)
        
        # Calculate MSE
        mse = np.mean((preds[1:] - sta) ** 2)
        
        ar_results.append({
            'sample_idx': i,
            'mse': mse,
            'predictions': preds,
            'ground_truth': sta,
            'inputs': xs,
        })
        logger.info(f"  Sample {i}: MSE = {mse:.6f}")
    
    return {
        'test_results': test_results,
        'test_samples': test_samples,
        'ar_results': ar_results,
        'test_config': test_config,
    }


def train_and_test(config: TrainingConfig) -> Dict[str, Any]:
    """
    Complete pipeline: train model and then test it.
    
    Args:
        config: Training configuration
        
    Returns:
        Dictionary with training and test results
    """
    results = {}
    
    # Train
    model, output_dir = train_model(config)
    results['model'] = model
    results['output_dir'] = output_dir
    
    # Test
    test_results = test_model(model, config)
    results.update(test_results)
    
    return results


def parse_args() -> TrainingConfig:
    """Parse command line arguments and return a TrainingConfig object."""
    parser = argparse.ArgumentParser(
        description='Train and test transformer on LTI trajectories'
    )
    
    # Training data
    parser.add_argument('--num_train_trajectories', type=int, default=20000)
    parser.add_argument('--num_val_trajectories', type=int, default=5000)
    parser.add_argument('--traj_length', type=int, default=50)
    parser.add_argument('--nx', type=int, default=10)
    parser.add_argument('--ny', type=int, default=5)
    parser.add_argument('--n_noise', type=int, default=1)
    parser.add_argument('--sigma_w', type=float, default=1e-1)
    parser.add_argument('--sigma_v', type=float, default=1e-1)
    parser.add_argument('--dataset_type', type=str, default='ypred',
                        choices=['ypred', 'noniid', 'upperTriA'])
    
    # Model
    parser.add_argument('--n_embd', type=int, default=256)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_positions', type=int, default=50)
    
    # Training settings
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)
    parser.add_argument('--gradient_clip_algorithm', type=str, default='norm')
    
    # Test settings
    parser.add_argument('--num_test_trajectories', type=int, default=100)
    parser.add_argument('--test_traj_length', type=int, default=50)
    parser.add_argument('--test_nx', type=int, default=None)
    parser.add_argument('--test_ny', type=int, default=None)
    parser.add_argument('--test_sigma_w', type=float, default=None)
    parser.add_argument('--test_sigma_v', type=float, default=None)
    parser.add_argument('--test_n_noise', type=int, default=None)
    
    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--ckpt_path', type=str, default=None)
    
    args = parser.parse_args()
    
    # Create config from args
    config = TrainingConfig()
    for key, value in vars(args).items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


if __name__ == "__main__":
    # Parse command line arguments
    config = parse_args()
    
    logger.info("=" * 80)
    logger.info("Starting LTI Transformer Training and Testing")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  Training: {config.num_train_trajectories} trajectories, length {config.traj_length}")
    logger.info(f"  State dim: {config.nx}, Obs dim: {config.ny}")
    logger.info(f"  Process noise: {config.sigma_w}, Obs noise: {config.sigma_v}")
    logger.info(f"  Model: GPT2 with {config.n_layer} layers, {config.n_embd} embedding")
    
    if config.test_sigma_w is not None or config.test_sigma_v is not None:
        logger.info(f"  Testing with DIFFERENT settings:")
        if config.test_sigma_w is not None:
            logger.info(f"    Process noise: {config.test_sigma_w}")
        if config.test_sigma_v is not None:
            logger.info(f"    Obs noise: {config.test_sigma_v}")
    
    # Run training and testing
    results = train_and_test(config)
    
    logger.info("\n" + "=" * 80)
    logger.info("Training and Testing Complete!")
    logger.info(f"Output saved to: {results['output_dir']}")
    logger.info("=" * 80)
