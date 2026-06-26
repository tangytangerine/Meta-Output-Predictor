#!/usr/bin/env python3
"""
Simplified Meta-Output Predictor Script

This script implements the core functionalities:
1. Generate training and validation trajectories with configurable system dynamics, noise, and input
2. Train a transformer model on those trajectories
3. Generate test trajectories with different parameters
4. Test the transformer on those test trajectories

This ignores the drone case and focuses on the LTI filtering system.

Usage:
    python simplified_script.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for scripts
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import argparse
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from transformers import MistralModel, MistralConfig

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================================
# 1. System Dynamics Model (LTI Filtering System)
# ============================================================================

class FilterSim:
    """
    LTI system simulator with configurable dynamics.
    
    Parameters:
    - nx: state dimension
    - ny: output dimension  
    - sigma_w: process noise std
    - sigma_v: measurement noise std
    - A: system matrix (if None, will be randomly generated)
    - C: output matrix (if None, will be randomly generated)
    """
    
    def __init__(self, nx=10, ny=5, sigma_w=0.1, sigma_v=0.1, A=None, C=None):
        self.nx = nx
        self.ny = ny
        self.sigma_w = sigma_w
        self.sigma_v = sigma_v
        
        # Generate random stable system matrix if not provided
        if A is None:
            self.A = self._generate_random_stable_matrix(nx)
        else:
            self.A = A
            
        # Generate random observability matrix if not provided
        if C is None:
            self.C = self._generate_observable_C(nx, ny, self.A)
        else:
            self.C = C
    
    @staticmethod
    def _generate_random_stable_matrix(n, eig_low=0.5, eig_high=0.8, max_tries=1000):
        """Generate a random stable matrix with eigenvalues in [eig_low, eig_high]"""
        for _ in range(max_tries):
            # Generate random eigenvalues in the stable region
            eigs = np.random.uniform(eig_low, eig_high, size=n)
            Lambda = np.diag(eigs)
            
            # Generate random invertible matrix
            P = np.random.randn(n, n)
            if np.linalg.matrix_rank(P) == n:
                A = P @ Lambda @ np.linalg.inv(P)
                # Check spectral radius for stability
                if np.max(np.abs(np.linalg.eigvals(A))) < 0.95:
                    return A
        
        # Fallback: diagonal matrix
        return np.diag(np.random.uniform(eig_low, eig_high, size=n))
    
    @staticmethod
    def _generate_observable_C(nx, ny, A):
        """Generate a random C matrix that ensures observability"""
        for _ in range(100):
            C = np.random.randn(ny, nx)
            # Check observability matrix rank
            O = np.concatenate([C @ np.linalg.matrix_power(A, i) 
                              for i in range(nx)], axis=0)
            if np.linalg.matrix_rank(O) == nx:
                return C.astype(np.float32)
        raise RuntimeError("Could not generate observable C matrix")
    
    def simulate(self, traj_len, x0=None, input_scale=1.0):
        """
        Simulate a trajectory.
        
        Args:
            traj_len: number of time steps
            x0: initial state (if None, random)
            input_scale: scale for control inputs
            
        Returns:
            states: (traj_len+1, nx) array of states
            outputs: (traj_len+1, ny) array of noisy observations
            inputs: (traj_len, nx) array of control inputs
        """
        nx, ny = self.nx, self.ny
        
        # Initialize
        x = np.random.randn(nx) if x0 is None else x0
        states = [x]
        outputs = [self.C @ x + np.random.randn(ny) * self.sigma_v]
        inputs = []
        
        for _ in range(traj_len):
            # Generate random control input (uniform in [0, 1] scaled by input_scale)
            u = np.random.uniform(0, 1, size=nx) * input_scale
            inputs.append(u)
            
            # State transition with process noise
            x = self.A @ x + u + np.random.randn(nx) * self.sigma_w
            states.append(x)
            
            # Observation with measurement noise
            y = self.C @ x + np.random.randn(ny) * self.sigma_v
            outputs.append(y)
        
        return (np.array(states, dtype=np.float32), 
                np.array(outputs, dtype=np.float32), 
                np.array(inputs, dtype=np.float32))


def generate_trajectory(nx=10, ny=5, traj_len=50, sigma_w=0.1, sigma_v=0.1, 
                         input_scale=1.0, A=None, C=None):
    """
    Generate a single trajectory with configurable parameters.
    
    Returns:
        dict with 'states', 'obs', 'inputs'
    """
    fsim = FilterSim(nx=nx, ny=ny, sigma_w=sigma_w, sigma_v=sigma_v, A=A, C=C)
    states, obs, inputs = fsim.simulate(traj_len, input_scale=input_scale)
    
    return {
        'states': states,
        'obs': obs,
        'inputs': inputs,
        'A': fsim.A,
        'C': fsim.C
    }


# ============================================================================
# 2. Dataset Preparation
# ============================================================================

class TrajectoryDataset(Dataset):
    """
    Dataset that provides trajectories for training.
    Each sample is a trajectory with states, observations, and inputs.
    """
    
    def __init__(self, trajectories):
        """
        Args:
            trajectories: list of trajectory dicts from generate_trajectory()
        """
        self.trajectories = trajectories
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        """
        Returns a sample dict with:
        - xs: input features (concatenated state and input)
        - ys: target (next state)
        """
        traj = self.trajectories[idx % len(self.trajectories)]
        
        states = traj['states']  # (traj_len+1, nx)
        inputs = traj['inputs']  # (traj_len, nx)
        
        # Input: [state_t, input_t], Target: state_{t+1}
        # Use true states as input (like original repository for LTI systems)
        xs = np.concatenate([states[:-1], inputs], axis=-1)  # (traj_len, nx+nx)
        ys = states[1:]  # (traj_len, nx)
        
        return {
            'xs': torch.from_numpy(xs).float(),
            'ys': torch.from_numpy(ys).float()
        }


def create_datasets(num_train=5000, num_val=500, traj_len=50, 
                    nx=10, ny=5, sigma_w=0.1, sigma_v=0.1, input_scale=1.0):
    """
    Create training and validation datasets.
    
    Returns:
        train_dataset, val_dataset
    """
    print("Generating training trajectories...")
    train_traj = [
        generate_trajectory(nx=nx, ny=ny, traj_len=traj_len,
                           sigma_w=sigma_w, sigma_v=sigma_v,
                           input_scale=input_scale)
        for _ in tqdm(range(num_train))
    ]
    
    print("Generating validation trajectories...")
    val_traj = [
        generate_trajectory(nx=nx, ny=ny, traj_len=traj_len,
                           sigma_w=sigma_w, sigma_v=sigma_v,
                           input_scale=input_scale)
        for _ in tqdm(range(num_val))
    ]
    
    return TrajectoryDataset(train_traj), TrajectoryDataset(val_traj)


# ============================================================================
# 3. Transformer Model
# ============================================================================

class LitTransformer(LightningModule):
    """
    Transformer model using HuggingFace Mistral for sequence prediction.
    Trained with PyTorch Lightning.
    Based on the original repository's approach.
    """
    
    def __init__(self, input_dim, output_dim, n_positions=50, 
                 n_embd=256, n_layer=8, n_head=6, dropout=0.15,
                 learning_rate=1e-4, weight_decay=1e-4, gradient_clip=1.0):
        """
        Args:
            input_dim: dimension of input features
            output_dim: dimension of output (state dimension)
            n_positions: maximum sequence length
            n_embd: embedding dimension
            n_layer: number of transformer layers
            n_head: number of attention heads
            dropout: dropout rate
            learning_rate: learning rate for optimizer
            weight_decay: weight decay for optimizer
            gradient_clip: gradient clipping value
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip
        
        # Input embedding (projects input features to embedding dimension)
        self.embed = nn.Linear(input_dim, n_embd)
        
        # Mistral configuration
        mistral_config = MistralConfig(
            vocab_size=1,  # Not used, we have our own embeddings
            max_position_embeddings=n_positions,
            hidden_size=n_embd,
            num_hidden_layers=n_layer,
            num_attention_heads=n_head,
            num_key_value_heads=n_head,  # Mistral uses GQA by default
            intermediate_size=n_embd * 4,  # Typical for Mistral
            hidden_dropout=dropout,
            attention_dropout=dropout,
            use_cache=False,
        )
        
        # Mistral model (without the final LM head)
        self.transformer = MistralModel(mistral_config)
        
        # Output projection
        self.output_proj = nn.Linear(n_embd, output_dim)
        
        # Loss function
        self.criterion = nn.MSELoss()
    
    def forward(self, xs):
        """
        Args:
            xs: (batch_size, seq_len, input_dim) input tensor
            
        Returns:
            preds: (batch_size, seq_len, output_dim) predictions
        """
        batch_size, seq_len, _ = xs.shape
        
        # Input embedding
        x = self.embed(xs)
        
        # Truncate if sequence is too long
        if seq_len > self.n_positions:
            x = x[:, -self.n_positions:]
        
        # Mistral expects inputs_embeds, not tokenizer input
        # Mistral has built-in positional embeddings and causal mask
        output = self.transformer(inputs_embeds=x).last_hidden_state
        
        # Output projection
        preds = self.output_proj(output)
        
        return preds
    
    def training_step(self, batch, batch_idx):
        xs = batch['xs']
        ys = batch['ys']
        
        preds = self(xs)
        
        # Handle potential length mismatch
        min_len = min(preds.shape[1], ys.shape[1])
        loss = self.criterion(preds[:, :min_len], ys[:, :min_len])
        
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        xs = batch['xs']
        ys = batch['ys']
        
        preds = self(xs)
        
        # Handle potential length mismatch
        min_len = min(preds.shape[1], ys.shape[1])
        loss = self.criterion(preds[:, :min_len], ys[:, :min_len])
        
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        xs = batch['xs']
        ys = batch['ys']
        
        preds = self(xs)
        
        # Handle potential length mismatch
        min_len = min(preds.shape[1], ys.shape[1])
        loss = self.criterion(preds[:, :min_len], ys[:, :min_len])
        
        self.log('test_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        return {
            'optimizer': optimizer,
            'clip_value': self.gradient_clip,
        }


# ============================================================================
# 4. Training Setup
# ============================================================================

class TransformerDataModule(pl.LightningDataModule):
    """Lightning data module for trajectory data."""
    
    def __init__(self, train_dataset, val_dataset=None, batch_size=32, num_workers=4):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )
    
    def val_dataloader(self):
        if self.val_dataset is not None:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False
            )
        return None
    
    def test_dataloader(self):
        if hasattr(self, 'test_dataset') and self.test_dataset is not None:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False
            )
        return None


# ============================================================================
# 5. Testing and Evaluation
# ============================================================================

def evaluate_model(model, dataset, batch_size=32):
    """
    Evaluate model on a dataset.
    
    Args:
        model: trained LitTransformer
        dataset: TrajectoryDataset to evaluate on
        batch_size: batch size for evaluation
        
    Returns:
        average MSE loss
    """
    model = model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.MSELoss()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in loader:
            xs = batch['xs'].to(device)
            ys = batch['ys'].to(device)
            
            preds = model(xs)
            min_len = min(preds.shape[1], ys.shape[1])
            loss = criterion(preds[:, :min_len], ys[:, :min_len])
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def generate_test_trajectories(num_test=100, traj_len=50, nx=10, ny=5,
                                sigma_w=0.2, sigma_v=0.2, input_scale=2.0):
    """
    Generate test trajectories with potentially different parameters than training.
    
    Args:
        num_test: number of test trajectories
        traj_len: length of each trajectory
        nx, ny: state and output dimensions
        sigma_w, sigma_v: noise parameters (can differ from training)
        input_scale: control input scale (can differ from training)
        
    Returns:
        TrajectoryDataset with test trajectories
    """
    print(f"\nGenerating {num_test} test trajectories with different parameters...")
    test_traj = [
        generate_trajectory(nx=nx, ny=ny, traj_len=traj_len,
                           sigma_w=sigma_w, sigma_v=sigma_v,
                           input_scale=input_scale)
        for _ in tqdm(range(num_test))
    ]
    return TrajectoryDataset(test_traj)


def plot_results(history, test_loss=None, save_path=None):
    """Plot training and validation loss."""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history and history['val_loss']:
        plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('Training History')
    
    if test_loss is not None:
        plt.subplot(1, 2, 2)
        plt.bar(['Test Loss'], [test_loss])
        plt.title('Test Performance')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        try:
            plt.show()
        except:
            # Non-interactive backend, skip show
            pass


def plot_trajectory_comparison(traj, model, config, save_path=None):
    """Plot comparison of true states vs predicted states."""
    states_true = traj['states']  # True states
    obs = traj['obs']              # Noisy observations
    inputs = traj['inputs']        # Control inputs
    
    # Prepare input for model (states[:-1] concatenated with inputs)
    xs_for_model = np.concatenate([states_true[:-1], inputs], axis=-1)
    xs_tensor = torch.from_numpy(xs_for_model).float().unsqueeze(0).to(device)
    
    # Ensure model is on the same device as input
    model = model.to(device)
    
    # Get model predictions
    model.eval()
    with torch.no_grad():
        preds_tensor = model(xs_tensor)
        
    preds = preds_tensor.squeeze(0).cpu().numpy()
    
    # Plot comparison
    plt.figure(figsize=(15, 10))
    
    # Plot first few state dimensions
    nx = config['nx']
    for i in range(min(4, nx)):
        plt.subplot(2, 2, i+1)
        plt.plot(states_true[1:, i], label='True State', linewidth=2)
        plt.plot(preds[:, i], '--', label='Predicted', linewidth=2)
        plt.plot(obs[:-1, i], ':', label='Observation', alpha=0.5, linewidth=1.5)
        plt.xlabel('Time Step')
        plt.ylabel(f'State Dimension {i}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.suptitle('True States vs Predicted States vs Noisy Observations', y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Trajectory plot saved to {save_path}")
    else:
        try:
            plt.show()
        except:
            # Non-interactive backend, skip show
            pass


# ============================================================================
# 6. Save and Load Model
# ============================================================================

def save_model(model, path='meta_output_predictor.pth', config=None):
    """Save the trained model."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, path)
    print(f"Model saved to {path}")


def load_model(path='meta_output_predictor.pth', config=None):
    """Load a saved model."""
    checkpoint = torch.load(path, map_location=device)
    
    if config is None:
        config = checkpoint.get('config', {})
    
    input_dim = config.get('nx', 10) + config.get('nx', 10)
    output_dim = config.get('nx', 10)
    
    model = LitTransformer(
        input_dim=input_dim,
        output_dim=output_dim,
        n_positions=config.get('n_positions', 50),
        n_embd=config.get('n_embd', 256),
        n_layer=config.get('n_layer', 8),
        n_head=config.get('n_head', 6),
        dropout=config.get('dropout', 0.15),
        learning_rate=config.get('learning_rate', 1e-4),
        weight_decay=config.get('weight_decay', 1e-4),
        gradient_clip=config.get('gradient_clip', 1.0)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"Model loaded from {path}")
    return model, config


# ============================================================================
# 7. Main Function
# ============================================================================

def main(args):
    # Configuration parameters (can be overridden by command line args)
    CONFIG = {
        # System dimensions
        'nx': args.nx if hasattr(args, 'nx') else 10,           # State dimension
        'ny': args.ny if hasattr(args, 'ny') else 5,            # Output dimension
        
        # Trajectory parameters
        'traj_len': args.traj_len if hasattr(args, 'traj_len') else 50,     # Length of each trajectory
        'num_train': args.num_train if hasattr(args, 'num_train') else 20000,  # Number of training trajectories
        'num_val': args.num_val if hasattr(args, 'num_val') else 5000,     # Number of validation trajectories
        'num_test': args.num_test if hasattr(args, 'num_test') else 100,    # Number of test trajectories
        
        # Noise parameters (training)
        'sigma_w_train': args.sigma_w_train if hasattr(args, 'sigma_w_train') else 0.01,  # Process noise
        'sigma_v_train': args.sigma_v_train if hasattr(args, 'sigma_v_train') else 0.01,  # Measurement noise
        'input_scale_train': args.input_scale_train if hasattr(args, 'input_scale_train') else 0.5,
        
        # Noise parameters (testing) - can be different!
        'sigma_w_test': args.sigma_w_test if hasattr(args, 'sigma_w_test') else 0.05,    # Higher process noise for testing
        'sigma_v_test': args.sigma_v_test if hasattr(args, 'sigma_v_test') else 0.05,    # Higher measurement noise for testing
        'input_scale_test': args.input_scale_test if hasattr(args, 'input_scale_test') else 1.0, # Different input scale for testing
        
        # Model parameters
        'n_positions': args.n_positions if hasattr(args, 'n_positions') else 50,   # Maximum sequence length
        'n_embd': args.n_embd if hasattr(args, 'n_embd') else 256,        # Embedding dimension
        'n_layer': args.n_layer if hasattr(args, 'n_layer') else 8,         # Number of transformer layers
        'n_head': args.n_head if hasattr(args, 'n_head') else 6,          # Number of attention heads
        'dropout': args.dropout if hasattr(args, 'dropout') else 0.15,       # Dropout rate
        
        # Training parameters
        'batch_size': args.batch_size if hasattr(args, 'batch_size') else 32,
        'num_epochs': args.num_epochs if hasattr(args, 'num_epochs') else 150,
        'learning_rate': args.learning_rate if hasattr(args, 'learning_rate') else 1e-4,
        'weight_decay': args.weight_decay if hasattr(args, 'weight_decay') else 1e-4,
        'gradient_clip': args.gradient_clip if hasattr(args, 'gradient_clip') else 1.0,
        
        # Output paths
        'save_model': args.save_model if hasattr(args, 'save_model') else None,
        'save_plot': args.save_plot if hasattr(args, 'save_plot') else 'results.png',
    }

    print("="*60)
    print("SIMPLE META-OUTPUT PREDICTOR")
    print("="*60)
    print("Configuration:")
    for key, value in CONFIG.items():
        if not key.startswith('save_'):
            print(f"  {key}: {value}")
    print()

    # Step 1: Create datasets
    print("\n[1/4] Creating datasets...")
    train_dataset, val_dataset = create_datasets(
        num_train=CONFIG['num_train'],
        num_val=CONFIG['num_val'],
        traj_len=CONFIG['traj_len'],
        nx=CONFIG['nx'],
        ny=CONFIG['ny'],
        sigma_w=CONFIG['sigma_w_train'],
        sigma_v=CONFIG['sigma_v_train'],
        input_scale=CONFIG['input_scale_train']
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, 
                              batch_size=CONFIG['batch_size'],
                              shuffle=True)
    val_loader = DataLoader(val_dataset, 
                            batch_size=CONFIG['batch_size'],
                            shuffle=False)

    print(f"Train dataset: {len(train_dataset)} trajectories")
    print(f"Val dataset: {len(val_dataset)} trajectories")

    # Step 2: Create model
    print("\n[2/4] Creating model...")
    input_dim = CONFIG['nx'] + CONFIG['nx']  # state + input dimensions
    output_dim = CONFIG['nx']  # predict next state

    model = LitTransformer(
        input_dim=input_dim,
        output_dim=output_dim,
        n_positions=CONFIG['n_positions'],
        n_embd=CONFIG['n_embd'],
        n_layer=CONFIG['n_layer'],
        n_head=CONFIG['n_head'],
        dropout=CONFIG['dropout'],
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        gradient_clip=CONFIG['gradient_clip']
    )

    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Step 3: Train the model with PyTorch Lightning
    print("\n[3/4] Training model with PyTorch Lightning...")
    
    # Create data module
    datamodule = TransformerDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=CONFIG['batch_size'],
        num_workers=4
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=CONFIG['num_epochs'],
        devices="auto",
        accelerator="auto",
        gradient_clip_val=CONFIG['gradient_clip'],
        log_every_n_steps=10,
        enable_progress_bar=True,
        default_root_dir=".",
    )
    
    # Train
    trainer.fit(model, datamodule=datamodule)
    
    # Get training history from Lightning logger
    # Access the logged metrics
    history = {
        'train_loss': [],
        'val_loss': []
    }
    if hasattr(trainer, 'logger') and trainer.logger is not None:
        # Try to extract metrics from logger
        pass  # Metrics are logged automatically by Lightning
    print(f"Training complete. Best val loss: {trainer.callback_metrics.get('val_loss', float('inf')):.6f}")
    
    # Step 4: Test with different parameters
    print("\n[4/4] Testing with different parameters...")
    test_dataset = generate_test_trajectories(
        num_test=CONFIG['num_test'],
        traj_len=CONFIG['traj_len'],
        nx=CONFIG['nx'],
        ny=CONFIG['ny'],
        sigma_w=CONFIG['sigma_w_test'],
        sigma_v=CONFIG['sigma_v_test'],
        input_scale=CONFIG['input_scale_test']
    )
    
    # Set up test datamodule
    datamodule.test_dataset = test_dataset
    test_results = trainer.test(model, datamodule=datamodule)
    test_loss = test_results[0]['test_loss']
    print(f"\nTest Loss (with different parameters): {test_loss:.6f}")

    # Plot results
    print("\nGenerating plots...")
    if CONFIG['save_plot']:
        plot_results(history, test_loss=test_loss, save_path=CONFIG['save_plot'])
    else:
        plot_results(history, test_loss=test_loss)

    # Generate and plot a single trajectory comparison
    print("\nGenerating single trajectory visualization...")
    test_traj = generate_trajectory(
        nx=CONFIG['nx'], 
        ny=CONFIG['ny'], 
        traj_len=CONFIG['traj_len'],
        sigma_w=CONFIG['sigma_w_test'],
        sigma_v=CONFIG['sigma_v_test'],
        input_scale=CONFIG['input_scale_test']
    )
    
    # Move model to eval mode for inference
    model.eval()
    
    if CONFIG['save_plot']:
        traj_plot_path = CONFIG['save_plot'].replace('.png', '_traj.png')
        plot_trajectory_comparison(test_traj, model, CONFIG, save_path=traj_plot_path)
    else:
        plot_trajectory_comparison(test_traj, model, CONFIG)

    # Compute MSE for this trajectory
    model = model.to(device)
    with torch.no_grad():
        xs_for_mse = torch.from_numpy(
            np.concatenate([test_traj['states'][:-1], test_traj['inputs']], axis=-1)
        ).float().unsqueeze(0).to(device)
        preds_for_mse = model(xs_for_mse).squeeze(0).cpu()
    # Account for model truncation to n_positions
    actual_pred_len = preds_for_mse.shape[0]
    actual_state_len = min(actual_pred_len, test_traj['states'][1:].shape[0])
    mse = np.mean((test_traj['states'][1:actual_state_len+1] - preds_for_mse.numpy()[:actual_state_len]) ** 2)
    print(f"\nSingle trajectory MSE: {mse:.6f}")
    print(f"Observation noise level (sigma_v): {CONFIG['sigma_v_test']:.2f}")
    print(f"Process noise level (sigma_w): {CONFIG['sigma_w_test']:.2f}")

    # Save model if requested
    if CONFIG['save_model']:
        save_model(model, CONFIG['save_model'], config=CONFIG)

    print("\n" + "="*60)
    print("Training and evaluation complete!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simplified Meta-Output Predictor')
    
    # System parameters
    parser.add_argument('--nx', type=int, default=10, help='State dimension')
    parser.add_argument('--ny', type=int, default=5, help='Output dimension')
    
    # Dataset parameters
    parser.add_argument('--traj_len', type=int, default=50, help='Trajectory length')
    parser.add_argument('--num_train', type=int, default=10000, help='Number of training trajectories')
    parser.add_argument('--num_val', type=int, default=2500, help='Number of validation trajectories')
    parser.add_argument('--num_test', type=int, default=100, help='Number of test trajectories')
    
    # Training noise parameters
    parser.add_argument('--sigma_w_train', type=float, default=0.01, help='Training process noise')
    parser.add_argument('--sigma_v_train', type=float, default=0.01, help='Training measurement noise')
    parser.add_argument('--input_scale_train', type=float, default=0.5, help='Training input scale')
    
    # Test noise parameters (can differ from training)
    parser.add_argument('--sigma_w_test', type=float, default=0.01, help='Test process noise')
    parser.add_argument('--sigma_v_test', type=float, default=0.01, help='Test measurement noise')
    parser.add_argument('--input_scale_test', type=float, default=1.0, help='Test input scale')
    
    # Model parameters
    parser.add_argument('--n_positions', type=int, default=50, help='Maximum sequence length')
    parser.add_argument('--n_embd', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--n_layer', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--n_head', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.15, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--gradient_clip', type=float, default=1.0, help='Gradient clipping')
    
    # Output
    parser.add_argument('--save_model', type=str, default=None, help='Path to save model')
    parser.add_argument('--save_plot', type=str, default='results.png', help='Path to save plots (default: results.png)')
    
    args = parser.parse_args()
    main(args)
