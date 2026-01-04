
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from src.modules import SlotVideoModel
import json
import numpy as np
from tqdm import tqdm

def analyze_ablation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    test_data = torch.load('datasets/bouncing_balls/test.pt')
    test_loader = DataLoader(TensorDataset(test_data), batch_size=4, shuffle=False)
    
    # Full Model
    model_full = SlotVideoModel(num_slots=6, slot_dim=64, use_interaction=True).to(device)
    model_full.load_state_dict(torch.load('results/models/slot_model.pth'))
    model_full.eval()
    
    # Ablated Model
    model_no_int = SlotVideoModel(num_slots=6, slot_dim=64, use_interaction=False).to(device)
    model_no_int.load_state_dict(torch.load('results/models/slot_model_no_interaction.pth'))
    model_no_int.eval()
    
    criterion = nn.MSELoss(reduction='none')
    
    mses_full = []
    mses_no_int = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Comparing"):
            video = batch[0].to(device)
            input_seq = video[:, :-1]
            target_seq = video[:, 1:]
            
            # Full
            out_full = model_full(input_seq)
            loss_full = criterion(out_full['pred_next_img'], target_seq)
            mses_full.extend(loss_full.mean(dim=(1, 2, 3, 4)).cpu().numpy())
            
            # No Interaction
            out_no_int = model_no_int(input_seq)
            loss_no_int = criterion(out_no_int['pred_next_img'], target_seq)
            mses_no_int.extend(loss_no_int.mean(dim=(1, 2, 3, 4)).cpu().numpy())
            
    mean_full = np.mean(mses_full)
    mean_no_int = np.mean(mses_no_int)
    
    print(f"Full Model MSE:          {mean_full:.6f}")
    print(f"No Interaction Model MSE: {mean_no_int:.6f}")
    
    improvement = (mean_no_int - mean_full) / mean_no_int * 100
    print(f"Improvement from Interaction: {improvement:.2f}%")
    
    with open('results/ablation_metrics.txt', 'w') as f:
        f.write(f"Full Model MSE: {mean_full:.6f}\n")
        f.write(f"No Interaction MSE: {mean_no_int:.6f}\n")
        f.write(f"Improvement: {improvement:.2f}%\n")

if __name__ == '__main__':
    analyze_ablation()
