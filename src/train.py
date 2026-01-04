
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from src.modules import SlotVideoModel
from src.baseline import ConvPredictor
from tqdm import tqdm

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Data
    print("Loading data...")
    train_data = torch.load('datasets/bouncing_balls/train.pt')
    test_data = torch.load('datasets/bouncing_balls/test.pt')
    
    # Normalize to 0-1 if not already (data gen put colors 0-1, but better check)
    # data_gen output is float32, colors 0-1.
    
    train_loader = DataLoader(TensorDataset(train_data), batch_size=args['batch_size'], shuffle=True)
    test_loader = DataLoader(TensorDataset(test_data), batch_size=args['batch_size'], shuffle=False)
    
    # Init Models
    slot_model = SlotVideoModel(num_slots=args['num_slots'], slot_dim=args['slot_dim']).to(device)
    baseline_model = ConvPredictor(hidden_dim=32).to(device)
    
    optimizer_slot = optim.Adam(slot_model.parameters(), lr=args['lr'])
    optimizer_base = optim.Adam(baseline_model.parameters(), lr=args['lr'])
    
    history = {'slot': [], 'baseline': []}
    
    os.makedirs('results/models', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    
    criterion = nn.MSELoss()
    
    print("Starting training...")
    for epoch in range(args['epochs']):
        slot_model.train()
        baseline_model.train()
        torch.cuda.empty_cache()
        
        slot_loss_epoch = 0
        base_loss_epoch = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args['epochs']}"):
            video = batch[0].to(device) # (B, T, C, H, W)
            video = video[:, :10] # Slice to 10 frames to save memory
            
            # Prepare targets
            # Input: frames 0 to T-1
            # Target: frames 1 to T
            input_seq = video[:, :-1]
            target_seq = video[:, 1:]
            
            # --- Train Slot Model ---
            optimizer_slot.zero_grad()
            out = slot_model(input_seq)
            
            # Loss 1: Reconstruction of current frame (to learn objects)
            loss_recon = criterion(out['recon_img'], input_seq)
            
            # Loss 2: Prediction of next frame (to learn dynamics)
            loss_pred = criterion(out['pred_next_img'], target_seq)
            
            loss_slot = loss_recon + loss_pred
            loss_slot.backward()
            optimizer_slot.step()
            slot_loss_epoch += loss_slot.item()
            
            # --- Train Baseline ---
            optimizer_base.zero_grad()
            pred_base = baseline_model(input_seq)
            loss_base = criterion(pred_base, target_seq)
            loss_base.backward()
            optimizer_base.step()
            base_loss_epoch += loss_base.item()
            
        # Validation
        avg_slot_loss = slot_loss_epoch / len(train_loader)
        avg_base_loss = base_loss_epoch / len(train_loader)
        
        history['slot'].append(avg_slot_loss)
        history['baseline'].append(avg_base_loss)
        
        print(f"Epoch {epoch+1}: Slot Loss={avg_slot_loss:.4f}, Base Loss={avg_base_loss:.4f}")
        
    # Save Models
    torch.save(slot_model.state_dict(), 'results/models/slot_model.pth')
    torch.save(baseline_model.state_dict(), 'results/models/baseline_model.pth')
    
    # Save History
    with open('results/metrics.json', 'w') as f:
        json.dump(history, f)
        
    # Generate Plots
    plt.figure()
    plt.plot(history['slot'], label='Slot Model')
    plt.plot(history['baseline'], label='Baseline')
    plt.legend()
    plt.title('Training Loss')
    plt.savefig('results/plots/training_curve.png')
    
    # Visual Evaluation on Test Set
    slot_model.eval()
    baseline_model.eval()
    
    with torch.no_grad():
        test_video = test_data[:2].to(device) # Take 2 examples
        input_seq = test_video[:, :-1]
        target_seq = test_video[:, 1:]
        
        print(f"Test input shape: {input_seq.shape}")
        out_slot = slot_model(input_seq)
        out_base = baseline_model(input_seq)
        print(f"Slot output shape: {out_slot['pred_next_img'].shape}")
        
        # Plot Frame 0 -> Pred Frame 1
        # We visualize the LAST transition in the sequence to see if it holds up
        t_idx = input_seq.shape[1] - 1
        
        fig, axs = plt.subplots(2, 5, figsize=(15, 6))
        # Cols: Input(t), Target(t+1), BasePred(t+1), SlotPred(t+1), SlotMasks(t)
        
        for i in range(2):
            # Input
            axs[i, 0].imshow(input_seq[i, t_idx].permute(1, 2, 0).cpu().numpy())
            axs[i, 0].set_title(f"Input t={t_idx}")
            axs[i, 0].axis('off')
            
            # Target
            axs[i, 1].imshow(target_seq[i, t_idx].permute(1, 2, 0).cpu().numpy())
            axs[i, 1].set_title(f"Target t={t_idx+1}")
            axs[i, 1].axis('off')
            
            # Baseline Pred
            axs[i, 2].imshow(out_base[i, t_idx].permute(1, 2, 0).cpu().numpy())
            axs[i, 2].set_title("Baseline Pred")
            axs[i, 2].axis('off')
            
            # Slot Pred
            axs[i, 3].imshow(out_slot['pred_next_img'][i, t_idx].permute(1, 2, 0).cpu().numpy())
            axs[i, 3].set_title("Slot Pred")
            axs[i, 3].axis('off')
            
            # Slot Masks (Combine masks into one image)
            # masks: (B, T, N, 1, H, W)
            masks = out_slot['masks'][i, t_idx] # (N, 1, H, W)
            # Create RGB visualization of masks
            # Assign color to each slot
            mask_viz = torch.zeros(3, 64, 64).to(device)
            colors = torch.tensor([[1,0,0], [0,1,0], [0,0,1], [1,1,0], [0,1,1], [1,0,1]]).to(device)
            
            for k in range(min(masks.shape[0], 6)):
                m = masks[k] # (1, H, W)
                c = colors[k].view(3, 1, 1)
                mask_viz += m * c
                
            axs[i, 4].imshow(mask_viz.permute(1, 2, 0).cpu().numpy())
            axs[i, 4].set_title("Slot Attention")
            axs[i, 4].axis('off')
            
        plt.tight_layout()
        plt.savefig('results/plots/qualitative_comparison.png')
        print("Plots saved.")

if __name__ == '__main__':
    args = {
        'batch_size': 4,
        'lr': 0.0004,
        'epochs': 20, # Short training for session
        'num_slots': 6, # Match max balls
        'slot_dim': 64
    }
    train(args)
