
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import json
from src.modules import SlotVideoModel
from tqdm import tqdm

def train_ablation(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Data
    print("Loading data...")
    train_data = torch.load('datasets/bouncing_balls/train.pt')
    
    train_loader = DataLoader(TensorDataset(train_data), batch_size=args['batch_size'], shuffle=True)
    
    # Init Model NO INTERACTION
    model = SlotVideoModel(num_slots=args['num_slots'], slot_dim=args['slot_dim'], use_interaction=False).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    
    os.makedirs('results/models', exist_ok=True)
    
    criterion = nn.MSELoss()
    
    print("Starting ablation training (No Interaction)...")
    for epoch in range(args['epochs']):
        model.train()
        torch.cuda.empty_cache()
        
        loss_epoch = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args['epochs']}"):
            video = batch[0].to(device) # (B, T, C, H, W)
            video = video[:, :10] # Slice to 10 frames
            
            input_seq = video[:, :-1]
            target_seq = video[:, 1:]
            
            optimizer.zero_grad()
            out = model(input_seq)
            
            loss_recon = criterion(out['recon_img'], input_seq)
            loss_pred = criterion(out['pred_next_img'], target_seq)
            
            loss = loss_recon + loss_pred
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
            
        avg_loss = loss_epoch / len(train_loader)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}")
        
    # Save Model
    torch.save(model.state_dict(), 'results/models/slot_model_no_interaction.pth')
    print("Ablation model saved.")

if __name__ == '__main__':
    args = {
        'batch_size': 4,
        'lr': 0.0004,
        'epochs': 20,
        'num_slots': 6,
        'slot_dim': 64
    }
    train_ablation(args)
