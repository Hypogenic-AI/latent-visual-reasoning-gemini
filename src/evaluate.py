
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from src.modules import SlotVideoModel
from src.baseline import ConvPredictor
import json
import numpy as np
from tqdm import tqdm

def evaluate_full():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    test_data = torch.load('datasets/bouncing_balls/test.pt')
    test_loader = DataLoader(TensorDataset(test_data), batch_size=4, shuffle=False)
    
    # Init Models
    slot_model = SlotVideoModel(num_slots=6, slot_dim=64).to(device)
    baseline_model = ConvPredictor(hidden_dim=32).to(device)
    
    slot_model.load_state_dict(torch.load('results/models/slot_model.pth'))
    baseline_model.load_state_dict(torch.load('results/models/baseline_model.pth'))
    
    slot_model.eval()
    baseline_model.eval()
    
    criterion = nn.MSELoss(reduction='none') # To compute std
    
    slot_mses = []
    base_mses = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            video = batch[0].to(device)
            input_seq = video[:, :-1]
            target_seq = video[:, 1:]
            
            # Slot
            out_slot = slot_model(input_seq)
            mse_slot = criterion(out_slot['pred_next_img'], target_seq)
            # Mean over pixels/time, keep batch dim
            mse_slot = mse_slot.mean(dim=(1, 2, 3, 4))
            slot_mses.extend(mse_slot.cpu().numpy())
            
            # Base
            out_base = baseline_model(input_seq)
            mse_base = criterion(out_base, target_seq)
            mse_base = mse_base.mean(dim=(1, 2, 3, 4))
            base_mses.extend(mse_base.cpu().numpy())
            
    mean_slot_mse = np.mean(slot_mses)
    std_slot_mse = np.std(slot_mses)
    mean_base_mse = np.mean(base_mses)
    std_base_mse = np.std(base_mses)
    
    print(f"Slot Model MSE: {mean_slot_mse:.6f} +/- {std_slot_mse:.6f}")
    print(f"Baseline MSE:   {mean_base_mse:.6f} +/- {std_base_mse:.6f}")
    
    results = {
        'slot_mse_mean': float(mean_slot_mse),
        'slot_mse_std': float(std_slot_mse),
        'base_mse_mean': float(mean_base_mse),
        'base_mse_std': float(std_base_mse)
    }
    
    with open('results/test_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    evaluate_full()
