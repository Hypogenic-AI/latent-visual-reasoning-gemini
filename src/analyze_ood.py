
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from src.modules import SlotVideoModel
from src.baseline import ConvPredictor
import json
import numpy as np
from tqdm import tqdm

def analyze_ood():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load OOD Data
    test_data = torch.load('datasets/bouncing_balls_ood/test_6balls.pt')
    test_loader = DataLoader(TensorDataset(test_data), batch_size=4, shuffle=False)
    
    # Load Models (Trained on 3 balls)
    slot_model = SlotVideoModel(num_slots=6, slot_dim=64).to(device)
    baseline_model = ConvPredictor(hidden_dim=32).to(device)
    
    slot_model.load_state_dict(torch.load('results/models/slot_model.pth'))
    baseline_model.load_state_dict(torch.load('results/models/baseline_model.pth'))
    
    slot_model.eval()
    baseline_model.eval()
    
    criterion = nn.MSELoss(reduction='none')
    
    mses_slot = []
    mses_base = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating OOD"):
            video = batch[0].to(device)
            input_seq = video[:, :-1]
            target_seq = video[:, 1:]
            
            # Slot Model
            out_slot = slot_model(input_seq)
            loss_slot = criterion(out_slot['pred_next_img'], target_seq)
            mses_slot.extend(loss_slot.mean(dim=(1, 2, 3, 4)).cpu().numpy())
            
            # Baseline
            out_base = baseline_model(input_seq)
            loss_base = criterion(out_base, target_seq)
            mses_base.extend(loss_base.mean(dim=(1, 2, 3, 4)).cpu().numpy())
            
    mean_slot = np.mean(mses_slot)
    mean_base = np.mean(mses_base)
    
    print(f"OOD (6 balls) - Slot Model MSE: {mean_slot:.6f}")
    print(f"OOD (6 balls) - Baseline MSE:   {mean_base:.6f}")
    
    # Load ID metrics for comparison
    with open('results/test_metrics.json', 'r') as f:
        id_metrics = json.load(f)
        
    id_slot = id_metrics['slot_mse_mean']
    id_base = id_metrics['base_mse_mean']
    
    print("\n--- Generalization Gap ---")
    print(f"Slot: {id_slot:.6f} (ID) -> {mean_slot:.6f} (OOD) | Increase: {(mean_slot-id_slot)/id_slot*100:.1f}%")
    print(f"Base: {id_base:.6f} (ID) -> {mean_base:.6f} (OOD) | Increase: {(mean_base-id_base)/id_base*100:.1f}%")
    
    with open('results/ood_metrics.txt', 'w') as f:
        f.write(f"Slot OOD MSE: {mean_slot:.6f}\n")
        f.write(f"Base OOD MSE: {mean_base:.6f}\n")

if __name__ == '__main__':
    analyze_ood()
