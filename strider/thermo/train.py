import argparse
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strider.thermo.differentiable import ThermoParameters, BatchedPartitionFunction

class RNADataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, "r") as f:
            self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return {
            "sequence": self.data[idx]["sequence"],
            "energy": self.data[idx]["energy"]
        }

def collate_fn(batch):
    sequences = [item["sequence"] for item in batch]
    energies = torch.tensor([item["energy"] for item in batch], dtype=torch.float32)
    return sequences, energies

def main():
    parser = argparse.ArgumentParser(description="Train differentiable Strider")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    args = parser.parse_args()

    print("Initializing Batched Differentiable Strider...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load the ArchiveII dataset
    data_dir = Path(__file__).parent.parent.parent / "data" / "datasets"
    train_path = data_dir / "archiveII_train.json"
    val_path = data_dir / "archiveII_val.json"
    if not train_path.exists():
        print("Please run scripts/download_datasets.py first to generate the dataset.")
        return

    train_dataset = RNADataset(train_path)
    val_dataset = RNADataset(val_path)
    
    BATCH_SIZE = 32
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    print(f"Loaded {len(train_dataset)} train / {len(val_dataset)} val examples from ArchiveII.")
    
    # 2. Setup model
    params = ThermoParameters(material="rna").to(device)
    model = BatchedPartitionFunction(params).to(device)
    
    # Store original parameters for comparison
    true_stack = params.stack_table.clone().detach()
    true_term = params.term_mismatch.clone().detach()
    true_hp = params.hairpin_sizes.clone().detach()
    true_ml_base = torch.exp(params.ml_base_raw).clone().detach()
    true_ml_init = torch.exp(params.ml_init_raw).clone().detach()
    true_ml_pair = torch.exp(params.ml_pair_raw).clone().detach()
    true_int = params.interior_sizes.clone().detach()
    true_bulge = torch.exp(params.bulge_sizes_raw).clone().detach()
    true_ninio = params.asymmetry_ninio.clone().detach()
    true_d3 = params.dangle_3.clone().detach()
    true_d5 = params.dangle_5.clone().detach()
    
    # Perturb the parameters to simulate "bad" initialization
    with torch.no_grad():
        params.stack_table += 1.0
        params.term_mismatch -= 0.5
        params.hairpin_sizes += 0.5
        params.ml_base_raw += 0.2
        params.ml_init_raw += 0.5
        params.ml_pair_raw += 0.2
        params.interior_sizes += 0.5
        params.bulge_sizes_raw += 0.5
        params.asymmetry_ninio += 0.2
        params.dangle_3 += 0.2
        params.dangle_5 += 0.2
        
    def param_mse_tensor():
        return (
            torch.nn.functional.mse_loss(params.stack_table, true_stack) +
            torch.nn.functional.mse_loss(params.term_mismatch, true_term) +
            torch.nn.functional.mse_loss(params.hairpin_sizes, true_hp) +
            torch.nn.functional.mse_loss(torch.exp(params.ml_base_raw), true_ml_base) +
            torch.nn.functional.mse_loss(torch.exp(params.ml_init_raw), true_ml_init) +
            torch.nn.functional.mse_loss(torch.exp(params.ml_pair_raw), true_ml_pair) +
            torch.nn.functional.mse_loss(params.interior_sizes, true_int) +
            torch.nn.functional.mse_loss(torch.exp(params.bulge_sizes_raw), true_bulge) +
            torch.nn.functional.mse_loss(params.asymmetry_ninio, true_ninio) +
            torch.nn.functional.mse_loss(params.dangle_3, true_d3) +
            torch.nn.functional.mse_loss(params.dangle_5, true_d5)
        )
        
    def param_mse():
        return param_mse_tensor().item()
        
    print(f"Total MSE of ALL parameters before training: {param_mse():.4f}")
    
    # 3. Setup optimizer
    # Separate structural parameters from neural parameters
    thermo_params = [v for k, v in model.named_parameters() if 'mlp' not in k]
    neural_params = [v for k, v in model.named_parameters() if 'mlp' in k]

    optimizer = optim.Adam([
        {'params': thermo_params, 'lr': 5e-2, 'weight_decay': 1e-5}, # Aggressive baseline physical learning
        {'params': neural_params, 'lr': 1e-3, 'weight_decay': 1e-4}  # Conservative neural fine-tuning
    ])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    criterion = nn.MSELoss()
    
    # 4. Training loop
    epochs = 25
    start_epoch = 0
    checkpoint_path = data_dir / "checkpoint.pt"
    
    if args.resume and checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Resuming from epoch {start_epoch + 1}")
    else:
        # Reset checkpoint if it exists to start fresh with new architecture
        if checkpoint_path.exists():
            print("Removing old checkpoint to start fresh...")
            checkpoint_path.unlink()
    
    print(f"\nStarting training for {epochs} epochs...")
    
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        
        model.train()
        total_loss = 0.0
        
        for sequences, true_energy in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            true_energy = true_energy.to(device).to(torch.float64)
            
            optimizer.zero_grad()
            pred_energy = model(sequences)
            loss = criterion(pred_energy, true_energy)
            # Gentle physical anchor to prevent parameter inflation
            reg_loss = 0.2 * param_mse_tensor()
            
            # Anchor global stacking array: increase L2 penalty on stack_table during final iterations
            if epoch >= epochs - 5:
                reg_loss += 0.5 * torch.sum((model.params.stack_table - true_stack)**2)
                
            total_batch_loss = loss + reg_loss
            
            total_batch_loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(sequences)
            
        avg_loss = total_loss / len(train_dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, true_energy in tqdm(val_loader, desc=f"Val {epoch+1}/{epochs}"):
                true_energy = true_energy.to(device).to(torch.float64)
                pred_energy = model(sequences)
                val_loss += criterion(pred_energy, true_energy).item() * len(sequences)
                
        avg_val_loss = val_loss / len(val_dataset)

        current_param_mse = param_mse()
        epoch_time = time.time() - epoch_start

        scheduler.step(avg_val_loss)
        print(f"Epoch [{epoch+1}/{epochs}] | Train RMSE: {avg_loss**0.5:.4f} kcal/mol | Val RMSE: {avg_val_loss**0.5:.4f} kcal/mol | Param MSE: {current_param_mse:.4f} | Time: {epoch_time:.2f}s")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_loss,
            'val_loss': avg_val_loss,
        }
        torch.save(checkpoint, checkpoint_path)
        
    print("\nTraining complete!")
    print("Gradient check on various parameters:")
    aauu = 0 * 64 + 0 * 16 + 3 * 4 + 3  # 5'-AA-3'/3'-UU-5' stack
    print(f"  AAUU stack -> True: {true_stack[aauu]:.4f}, Learned: {params.stack_table[aauu].item():.4f}")
    print(f"  ML Base -> True: {true_ml_base.item():.4f}, Learned: {torch.exp(params.ml_base_raw).item():.4f}")
    print(f"  Hairpin Size 4 -> True: {true_hp[4]:.4f}, Learned: {params.get_hairpin_size_energy(4).item():.4f}")
    print(f"  Interior Size 3 -> True: {true_int[2]:.4f}, Learned: {params.interior_sizes[2].item():.4f}")
    print(f"  Bulge Size 2 -> True: {true_bulge[1]:.4f}, Learned: {torch.exp(params.bulge_sizes_raw)[1].item():.4f}")

if __name__ == "__main__":
    main()
