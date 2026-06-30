#!/usr/bin/env python3
"""
Simplified Meta-Output Predictor Script with Mamba SSM

This script implements the core functionalities:
1. Generate training and validation trajectories with configurable system dynamics, noise, and input
2. Train a Mamba SSM model on those trajectories
3. Generate test trajectories with different parameters
4. Test the Mamba SSM on those test trajectories

This ignores the drone case and focuses on the LTI filtering system.

Usage:
    python simplified_script_mamba.py
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
from mamba_ssm import Mamba
from filterpy.kalman import KalmanFilter

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
    
    def __init__(self, nx=10, ny=5, sigma_w=0.1, sigma_v=0.1, A=None, C=None,
                 eig_low=0.5, eig_high=0.8, input_lower=0.0, input_upper=1.0):
        self.nx = nx
        self.ny = ny
        self.sigma_w = sigma_w
        self.sigma_v = sigma_v
        self.input_lower = input_lower
        self.input_upper = input_upper
        
        # Generate random stable system matrix if not provided
        if A is None:
            self.A = self._generate_random_stable_matrix(nx, eig_low=eig_low, eig_high=eig_high)
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
    
    def simulate(self, traj_len, x0=None, input_lower=None, input_upper=None):
        """
        Simulate a trajectory.
        
        Args:
            traj_len: number of time steps
            x0: initial state (if None, random)
            input_lower: lower bound for control inputs (if None, uses self.input_lower)
            input_upper: upper bound for control inputs (if None, uses self.input_upper)
            
        Returns:
            states: (traj_len+1, nx) array of states
            outputs: (traj_len+1, ny) array of noisy observations
            inputs: (traj_len, nx) array of control inputs
        """
        nx, ny = self.nx, self.ny
        
        # Use instance defaults if not provided
        if input_lower is None:
            input_lower = self.input_lower
        if input_upper is None:
            input_upper = self.input_upper
        
        # Initialize
        x = np.random.randn(nx) if x0 is None else x0
        states = [x]
        outputs = [self.C @ x + np.random.randn(ny) * self.sigma_v]
        inputs = []
        
        for _ in range(traj_len):
            # Generate random control input (uniform in [input_lower, input_upper])
            u = np.random.uniform(input_lower, input_upper, size=nx)
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


def apply_kalman_filter(fsim, obs, inputs=None, x0=None, P0=None, sigma_w=None, sigma_v=None):
    """
    Apply Kalman filter to a trajectory's observations with control inputs.
    
    Args:
        fsim: FilterSim instance with A and C matrices
        obs: (traj_len+1, ny) array of noisy observations
        inputs: (traj_len, nx) array of control inputs (if None, assumes no control)
        x0: initial state estimate (if None, uses zeros)
        P0: initial covariance (if None, uses identity)
        sigma_w: process noise (if None, uses fsim.sigma_w)
        sigma_v: measurement noise (if None, uses fsim.sigma_v)
    
    Returns:
        kf_states: (traj_len+1, nx) array of Kalman filter state estimates
    """
    ny, nx = fsim.C.shape
    
    sigma_w = fsim.sigma_w if sigma_w is None else sigma_w
    sigma_v = fsim.sigma_v if sigma_v is None else sigma_v

    kf = KalmanFilter(dim_x=nx, dim_z=ny)
    kf.Q = np.eye(nx) * sigma_w ** 2
    kf.R = np.eye(ny) * sigma_v ** 2
    kf.P = np.eye(nx) if P0 is None else P0
    kf.x = np.zeros(nx) if x0 is None else x0
    kf.F = fsim.A
    kf.H = fsim.C
    
    # Control matrix B is identity for this system (x_{t+1} = A @ x_t + u_t + w_t)
    kf.B = np.eye(nx)
    
    # Store predictions
    kf_states = [kf.x.copy()]
    
    for i, y in enumerate(obs[1:]):  # Start from second observation (first is initial state)
        # Standard Kalman filter loop: predict next state, then update with observation
        if inputs is not None and i < len(inputs):
            kf.predict(u=inputs[i])
        else:
            kf.predict()
        kf.update(y)
        kf_states.append(kf.x.copy())
    
    return np.array(kf_states, dtype=np.float32)


def apply_kalman_filter_from_traj(traj):
    """
    Apply Kalman filter to a trajectory dictionary.
    
    Args:
        traj: trajectory dictionary with 'obs', 'inputs', 'A', 'C' (and optionally 'sigma_w', 'sigma_v')
    
    Returns:
        kf_states: (traj_len+1, nx) array of Kalman filter state estimates
    """
    # Create a FilterSim instance with the trajectory's parameters
    nx = traj['A'].shape[0]
    ny = traj['C'].shape[0]
    
    # Get noise parameters from trajectory or use defaults
    sigma_w = traj.get('sigma_w', 0.1)  # Default to 0.1 if not present
    sigma_v = traj.get('sigma_v', 0.1)  # Default to 0.1 if not present
    
    fsim = FilterSim(nx=nx, ny=ny, sigma_w=sigma_w, sigma_v=sigma_v)
    fsim.A = traj['A']
    fsim.C = traj['C']
    
    return apply_kalman_filter(fsim, traj['obs'], traj['inputs'])


def generate_trajectory(nx=10, ny=5, traj_len=50, sigma_w=0.1, sigma_v=0.1, 
                         input_lower=0.0, input_upper=1.0, A=None, C=None,
                         eig_low=0.5, eig_high=0.8):
    """
    Generate a single trajectory with configurable parameters.
    
    Returns:
        dict with 'states', 'obs', 'inputs', 'A', 'C', 'sigma_w', 'sigma_v'
    """
    fsim = FilterSim(nx=nx, ny=ny, sigma_w=sigma_w, sigma_v=sigma_v, A=A, C=C,
                     eig_low=eig_low, eig_high=eig_high,
                     input_lower=input_lower, input_upper=input_upper)
    states, obs, inputs = fsim.simulate(traj_len, input_lower=input_lower, input_upper=input_upper)
    
    return {
        'states': states,
        'obs': obs,
        'inputs': inputs,
        'A': fsim.A,
        'C': fsim.C,
        'sigma_w': sigma_w,
        'sigma_v': sigma_v
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
                    nx=10, ny=5, sigma_w=0.1, sigma_v=0.1, 
                    input_lower=0.0, input_upper=1.0,
                    eig_low=0.5, eig_high=0.8):
    """
    Create training and validation datasets.
    
    Returns:
        train_dataset, val_dataset
    """
    print("Generating training trajectories...")
    train_traj = [
        generate_trajectory(nx=nx, ny=ny, traj_len=traj_len,
                           sigma_w=sigma_w, sigma_v=sigma_v,
                           input_lower=input_lower, input_upper=input_upper,
                           eig_low=eig_low, eig_high=eig_high)
        for _ in tqdm(range(num_train))
    ]
    
    print("Generating validation trajectories...")
    val_traj = [
        generate_trajectory(nx=nx, ny=ny, traj_len=traj_len,
                           sigma_w=sigma_w, sigma_v=sigma_v,
                           input_lower=input_lower, input_upper=input_upper,
                           eig_low=eig_low, eig_high=eig_high)
        for _ in tqdm(range(num_val))
    ]
    
    return TrajectoryDataset(train_traj), TrajectoryDataset(val_traj)


# ============================================================================
# 3. Mamba SSM Model
# ============================================================================

class LitMamba(LightningModule):
    """
    Mamba SSM model for sequence prediction.
    Trained with PyTorch Lightning.
    Based on the original repository's approach but using Mamba instead of Transformer.
    """
    
    def __init__(self, input_dim, output_dim, n_positions=50, 
                 d_model=256, n_layers=4, d_state=16, d_conv=4, expand=2,
                 learning_rate=1e-4, weight_decay=1e-4, gradient_clip=1.0):
        """
        Args:
            input_dim: dimension of input features
            output_dim: dimension of output (state dimension)
            n_positions: maximum sequence length
            d_model: model dimension
            n_layers: number of Mamba layers
            d_state: state dimension for Mamba
            d_conv: convolution dimension for Mamba
            expand: expansion factor for Mamba
            learning_rate: learning rate for optimizer
            weight_decay: weight decay for optimizer
            gradient_clip: gradient clipping value
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_positions = n_positions
        self.d_model = d_model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip
        
        # Input projection to model dimension
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Mamba SSM layers
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            ) for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_model, output_dim)
        
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
        
        # Input projection
        x = self.input_proj(xs)  # (batch_size, seq_len, d_model)
        
        # Truncate if sequence is too long
        if seq_len > self.n_positions:
            x = x[:, -self.n_positions:]
        
        # Apply Mamba layers
        for layer in self.mamba_layers:
            x = layer(x)
        
        # Output projection
        preds = self.output_proj(x)
        
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

class SequenceDataModule(pl.LightningDataModule):
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
        model: trained LitMamba
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
                                sigma_w=0.2, sigma_v=0.2, 
                                input_lower=0.0, input_upper=2.0,
                                eig_low=0.5, eig_high=0.8):
    """
    Generate test trajectories with potentially different parameters than training.
    
    Args:
        num_test: number of test trajectories
        traj_len: length of each trajectory
        nx, ny: state and output dimensions
        sigma_w, sigma_v: noise parameters (can differ from training)
        input_lower, input_upper: control input range (can differ from training)
        eig_low, eig_high: eigenvalue range for system matrix (can differ from training)
        
    Returns:
        TrajectoryDataset with test trajectories
    """
    print(f"\nGenerating {num_test} test trajectories with different parameters...")
    test_traj = [
        generate_trajectory(nx=nx, ny=ny, traj_len=traj_len,
                           sigma_w=sigma_w, sigma_v=sigma_v,
                           input_lower=input_lower, input_upper=input_upper,
                           eig_low=eig_low, eig_high=eig_high)
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
    
    # Get Kalman filter predictions
    kf_states = apply_kalman_filter_from_traj(traj)
    kf_preds = kf_states[1:]  # KF predictions start from step 1
    
    # Compute per-time-step MSE
    actual_pred_len = preds.shape[0]
    actual_state_len = min(actual_pred_len, states_true[1:].shape[0])
    
    # Randomly select a state dimension to plot
    nx = config['nx']
    ny = config['ny']
    # For plotting, select a dimension that exists in observation space
    # so we can show the observation alongside the state
    # Select from dimensions that are observable (0 to ny-1)
    state_dim = np.random.randint(0, ny)
    
    # Store data for trajectory plot
    plot_data = {
        'states_true': states_true[1:, state_dim],
        'preds': preds[:, state_dim],
        'kf_preds': kf_preds[:, state_dim],
        'obs': obs[:-1, state_dim],
        'state_dim': state_dim,
        'traj_idx': 0
    }
    
    # Create Figure 1: Trajectory comparison plot
    plt.figure(figsize=(15, 5))
    plt.plot(plot_data['states_true'], label='True State', linewidth=2)
    plt.plot(plot_data['preds'], '--', label='Mamba Predicted', linewidth=2)
    plt.plot(plot_data['kf_preds'], ':', label='Kalman Filter', linewidth=2)
    plt.plot(plot_data['obs'], ':', label='Observation', alpha=0.5, linewidth=1.5)
    plt.xlabel('Time Step')
    plt.ylabel(f'State Dim {plot_data["state_dim"]}')
    plt.title(f'System 0, State {plot_data["state_dim"]}')
    plt.legend()
    plt.grid(True, alpha=0.3)
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
    plt.close()


# ============================================================================
# 6. Save and Load Model
# ============================================================================

def save_model(model, path='mamba_meta_output_predictor.pth', config=None):
    """Save the trained model."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, path)
    print(f"Model saved to {path}")


def load_model(path='mamba_meta_output_predictor.pth', config=None):
    """Load a saved model."""
    checkpoint = torch.load(path, map_location=device)
    
    if config is None:
        config = checkpoint.get('config', {})
    
    input_dim = config.get('nx', 10) + config.get('nx', 10)
    output_dim = config.get('nx', 10)
    
    model = LitMamba(
        input_dim=input_dim,
        output_dim=output_dim,
        n_positions=config.get('n_positions', 50),
        d_model=config.get('d_model', 256),
        n_layers=config.get('n_layers', 4),
        d_state=config.get('d_state', 16),
        d_conv=config.get('d_conv', 4),
        expand=config.get('expand', 2),
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
        'input_lower_train': args.input_lower_train if hasattr(args, 'input_lower_train') else 0.0,  # Input lower bound
        'input_upper_train': args.input_upper_train if hasattr(args, 'input_upper_train') else 1.0,  # Input upper bound
        'eig_low_train': args.eig_low_train if hasattr(args, 'eig_low_train') else 0.5,    # Eigenvalue lower bound
        'eig_high_train': args.eig_high_train if hasattr(args, 'eig_high_train') else 0.8,   # Eigenvalue upper bound
        
        # Noise parameters (testing) - can be different!
        'sigma_w_test': args.sigma_w_test if hasattr(args, 'sigma_w_test') else 0.05,    # Higher process noise for testing
        'sigma_v_test': args.sigma_v_test if hasattr(args, 'sigma_v_test') else 0.05,    # Higher measurement noise for testing
        'input_lower_test': args.input_lower_test if hasattr(args, 'input_lower_test') else 0.0,  # Input lower bound for testing
        'input_upper_test': args.input_upper_test if hasattr(args, 'input_upper_test') else 2.0,  # Input upper bound for testing
        'eig_low_test': args.eig_low_test if hasattr(args, 'eig_low_test') else 0.5,      # Eigenvalue lower bound for testing
        'eig_high_test': args.eig_high_test if hasattr(args, 'eig_high_test') else 0.8,    # Eigenvalue upper bound for testing
        
        # Model parameters
        'n_positions': args.n_positions if hasattr(args, 'n_positions') else 50,   # Maximum sequence length
        'd_model': args.d_model if hasattr(args, 'd_model') else 256,        # Model dimension
        'n_layers': args.n_layers if hasattr(args, 'n_layers') else 4,         # Number of Mamba layers
        'd_state': args.d_state if hasattr(args, 'd_state') else 16,          # State dimension for Mamba
        'd_conv': args.d_conv if hasattr(args, 'd_conv') else 4,           # Convolution dimension for Mamba
        'expand': args.expand if hasattr(args, 'expand') else 2,            # Expansion factor for Mamba
        
        # Training parameters
        'batch_size': args.batch_size if hasattr(args, 'batch_size') else 32,
        'num_epochs': args.num_epochs if hasattr(args, 'num_epochs') else 150,
        'learning_rate': args.learning_rate if hasattr(args, 'learning_rate') else 1e-4,
        'weight_decay': args.weight_decay if hasattr(args, 'weight_decay') else 1e-4,
        'gradient_clip': args.gradient_clip if hasattr(args, 'gradient_clip') else 1.0,
        
        # Output paths
        'save_model': args.save_model if hasattr(args, 'save_model') else None,
        'save_plot': args.save_plot if hasattr(args, 'save_plot') else 'results_mamba.png',
    }

    print("="*60)
    print("SIMPLE MAMBA META-OUTPUT PREDICTOR")
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
        input_lower=CONFIG['input_lower_train'],
        input_upper=CONFIG['input_upper_train'],
        eig_low=CONFIG['eig_low_train'],
        eig_high=CONFIG['eig_high_train']
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

    model = LitMamba(
        input_dim=input_dim,
        output_dim=output_dim,
        n_positions=CONFIG['n_positions'],
        d_model=CONFIG['d_model'],
        n_layers=CONFIG['n_layers'],
        d_state=CONFIG['d_state'],
        d_conv=CONFIG['d_conv'],
        expand=CONFIG['expand'],
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        gradient_clip=CONFIG['gradient_clip']
    )

    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Step 3: Train the model with PyTorch Lightning
    print("\n[3/4] Training model with PyTorch Lightning...")
    
    # Create data module
    datamodule = SequenceDataModule(
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
        input_lower=CONFIG['input_lower_test'],
        input_upper=CONFIG['input_upper_test'],
        eig_low=CONFIG['eig_low_test'],
        eig_high=CONFIG['eig_high_test']
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

    # Generate 1000 test trajectories for visualization
    print("\nGenerating 1000 test trajectories for visualization...")
    test_trajs = [
        generate_trajectory(
            nx=CONFIG['nx'], 
            ny=CONFIG['ny'], 
            traj_len=CONFIG['traj_len'],
            sigma_w=CONFIG['sigma_w_test'],
            sigma_v=CONFIG['sigma_v_test'],
            input_lower=CONFIG['input_lower_test'],
            input_upper=CONFIG['input_upper_test'],
            eig_low=CONFIG['eig_low_test'],
            eig_high=CONFIG['eig_high_test']
        )
        for _ in tqdm(range(1000))
    ]
    
    # Randomly sample 4 trajectory/state combinations for the trajectory plot
    np.random.seed(42)  # For reproducibility
    sampled_indices = np.random.choice(len(test_trajs), size=4, replace=False)
    
    # Store data for trajectory plot
    plot_data = []
    all_mses = []
    
    for idx, traj_idx in enumerate(sampled_indices):
        test_traj = test_trajs[traj_idx]
        states_true = test_traj['states']
        obs = test_traj['obs']
        inputs = test_traj['inputs']
        
        # Prepare input for model
        xs_for_model = np.concatenate([states_true[:-1], inputs], axis=-1)
        xs_tensor = torch.from_numpy(xs_for_model).float().unsqueeze(0).to(device)
        
        # Get model predictions
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            preds_tensor = model(xs_tensor)
        
        preds = preds_tensor.squeeze(0).cpu().numpy()
        
        # Get Kalman filter predictions
        kf_states = apply_kalman_filter_from_traj(test_traj)
        kf_preds = kf_states[1:]  # KF predictions start from step 1
        
        # Compute per-time-step MSE
        actual_pred_len = preds.shape[0]
        actual_state_len = min(actual_pred_len, states_true[1:].shape[0])
        
        # Randomly select a state dimension to plot
        nx = CONFIG['nx']
        ny = CONFIG['ny']
        # For plotting, select a dimension that exists in observation space
        # so we can show the observation alongside the state
        # Select from dimensions that are observable (0 to ny-1)
        state_dim = np.random.randint(0, ny)
        
        # Store data for trajectory plot
        plot_data.append({
            'states_true': states_true[1:, state_dim],
            'preds': preds[:, state_dim],
            'kf_preds': kf_preds[:, state_dim],
            'obs': obs[:-1, state_dim],
            'state_dim': state_dim,
            'traj_idx': traj_idx
        })
        
        # Compute average MSE for this trajectory
        mse = np.mean((states_true[1:actual_state_len+1] - preds[:actual_state_len]) ** 2)
        all_mses.append(mse)
    
    # Create Figure 1: Trajectory comparison plot (4 separate subplots)
    plt.figure(figsize=(15, 10))
    for idx, data in enumerate(plot_data):
        plt.subplot(2, 2, idx+1)
        plt.plot(data['states_true'], label='True State', linewidth=2)
        plt.plot(data['preds'], '--', label='Mamba Predicted', linewidth=2)
        plt.plot(data['kf_preds'], ':', label='Kalman Filter', linewidth=2)
        plt.plot(data['obs'], ':', label='Observation', alpha=0.5, linewidth=1.5)
        plt.xlabel('Time Step')
        plt.ylabel(f'State Dim {data["state_dim"]}')
        plt.title(f'System {data["traj_idx"]}, State {data["state_dim"]}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    plt.suptitle('True States vs Predicted States vs Kalman Filter vs Noisy Observations (Random Sample)', y=1.02)
    plt.tight_layout()
    if CONFIG['save_plot']:
        traj_plot_path = CONFIG['save_plot'].replace('.png', '_traj_mamba.png')
        plt.savefig(traj_plot_path)
        print(f"Trajectory plot saved to {traj_plot_path}")
    else:
        plt.show()
    plt.close()
    
    # Compute average MSE across the 4 sampled trajectories
    avg_mse_sampled = np.mean(all_mses)
    print(f"\nAverage MSE across 4 sampled trajectories: {avg_mse_sampled:.6f}")

    # Compute per-timestep average MSE across ALL 1000 trajectories
    print("\nComputing per-timestep average MSE across all 1000 trajectories...")
    all_per_timestep_mses = []
    all_kf_per_timestep_mses = []
    
    for test_traj in tqdm(test_trajs):
        states_true = test_traj['states']
        inputs = test_traj['inputs']
        
        # Get model predictions
        xs_for_model = np.concatenate([states_true[:-1], inputs], axis=-1)
        xs_tensor = torch.from_numpy(xs_for_model).float().unsqueeze(0).to(device)
        
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            preds_tensor = model(xs_tensor)
        
        preds = preds_tensor.squeeze(0).cpu().numpy()
        actual_pred_len = preds.shape[0]
        actual_state_len = min(actual_pred_len, states_true[1:].shape[0])
        
        # Get Kalman filter predictions
        kf_states = apply_kalman_filter_from_traj(test_traj)
        kf_preds = kf_states[1:]  # KF predictions start from step 1
        kf_pred_len = min(kf_preds.shape[0], states_true[1:].shape[0])
        
        # Compute per-timestep L2 norm across all state dimensions for this trajectory
        per_timestep_err = np.linalg.norm((states_true[1:actual_state_len+1] - preds[:actual_state_len]), axis=-1)
        all_per_timestep_mses.append(per_timestep_err)
        
        # Compute per-timestep L2 norm for Kalman filter
        kf_per_timestep_err = np.linalg.norm((states_true[1:kf_pred_len+1] - kf_preds[:kf_pred_len]), axis=-1)
        all_kf_per_timestep_mses.append(kf_per_timestep_err)
    
    # Compute average and std per-timestep error across all trajectories
    min_length = min(len(err_curve) for err_curve in all_per_timestep_mses)
    trimmed_errs = [err_curve[:min_length] for err_curve in all_per_timestep_mses]
    avg_per_timestep_err = np.mean(trimmed_errs, axis=0)
    std_per_timestep_err = np.std(trimmed_errs, axis=0)
    
    # Compute average and std per-timestep error for Kalman filter
    kf_min_length = min(len(err_curve) for err_curve in all_kf_per_timestep_mses)
    trimmed_kf_errs = [err_curve[:kf_min_length] for err_curve in all_kf_per_timestep_mses]
    avg_kf_per_timestep_err = np.mean(trimmed_kf_errs, axis=0)
    std_kf_per_timestep_err = np.std(trimmed_kf_errs, axis=0)
    
    # Compute overall average error across all trajectories (matching plot_errs: mean of sum across time)
    traj_errs_sum = np.array([err_curve.sum() for err_curve in all_per_timestep_mses])
    avg_err_all = traj_errs_sum.mean()
    
    kf_traj_errs_sum = np.array([err_curve.sum() for err_curve in all_kf_per_timestep_mses])
    avg_kf_err_all = kf_traj_errs_sum.mean()
    
    # Create Figure 2: Error vs time step plot - single averaged curve with error band
    plt.figure(figsize=(15, 5))
    plt.plot(avg_per_timestep_err, label='Mamba Error', linewidth=2, color='blue')
    plt.fill_between(
        range(min_length),
        avg_per_timestep_err - std_per_timestep_err,
        avg_per_timestep_err + std_per_timestep_err,
        color='blue',
        alpha=0.2,
        label='Mamba ±1 std. dev.'
    )
    plt.plot(avg_kf_per_timestep_err, label='Kalman Filter Error', linewidth=2, color='green')
    plt.fill_between(
        range(kf_min_length),
        avg_kf_per_timestep_err - std_kf_per_timestep_err,
        avg_kf_per_timestep_err + std_kf_per_timestep_err,
        color='green',
        alpha=0.2,
        label='Kalman Filter ±1 std. dev.'
    )
    plt.xlabel('Time Step')
    plt.ylabel('L2 Error')
    plt.title('Average L2 Error vs Time Step (Across All 1000 Trajectories)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if CONFIG['save_plot']:
        mse_plot_path = CONFIG['save_plot'].replace('.png', '_mse_vs_time_mamba.png')
        plt.savefig(mse_plot_path)
        print(f"MSE vs Time plot saved to {mse_plot_path}")
    else:
        plt.show()
    plt.close()
    
    print(f"\nAverage Mamba Error across ALL 1000 trajectories: {avg_err_all:.2f}")
    print(f"Average Kalman Filter Error across ALL 1000 trajectories: {avg_kf_err_all:.2f}")
    print(f"Observation noise level (sigma_v): {CONFIG['sigma_v_test']:.2f}")
    print(f"Process noise level (sigma_w): {CONFIG['sigma_w_test']:.2f}")

    # Save model if requested
    if CONFIG['save_model']:
        save_model(model, CONFIG['save_model'], config=CONFIG)

    print("\n" + "="*60)
    print("Training and evaluation complete!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simplified Meta-Output Predictor with Mamba')
    
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
    parser.add_argument('--input_lower_train', type=float, default=0.0, help='Training input lower bound')
    parser.add_argument('--input_upper_train', type=float, default=1.0, help='Training input upper bound')
    parser.add_argument('--eig_low_train', type=float, default=0.5, help='Training eigenvalue lower bound')
    parser.add_argument('--eig_high_train', type=float, default=0.8, help='Training eigenvalue upper bound')
    
    # Test noise parameters (can differ from training)
    parser.add_argument('--sigma_w_test', type=float, default=0.01, help='Test process noise')
    parser.add_argument('--sigma_v_test', type=float, default=0.01, help='Test measurement noise')
    parser.add_argument('--input_lower_test', type=float, default=0.0, help='Test input lower bound')
    parser.add_argument('--input_upper_test', type=float, default=2.0, help='Test input upper bound')
    parser.add_argument('--eig_low_test', type=float, default=0.5, help='Test eigenvalue lower bound')
    parser.add_argument('--eig_high_test', type=float, default=0.8, help='Test eigenvalue upper bound')
    
    # Model parameters
    parser.add_argument('--n_positions', type=int, default=50, help='Maximum sequence length')
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of Mamba layers')
    parser.add_argument('--d_state', type=int, default=16, help='State dimension for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='Convolution dimension for Mamba')
    parser.add_argument('--expand', type=int, default=2, help='Expansion factor for Mamba')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--gradient_clip', type=float, default=1.0, help='Gradient clipping')
    
    # Output
    parser.add_argument('--save_model', type=str, default=None, help='Path to save model')
    parser.add_argument('--save_plot', type=str, default='results_mamba.png', help='Path to save plots (default: results_mamba.png)')
    
    args = parser.parse_args()
    main(args)