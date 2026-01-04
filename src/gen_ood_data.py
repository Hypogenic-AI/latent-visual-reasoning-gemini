
import numpy as np
import torch
import os
from tqdm import tqdm

def generate_bouncing_balls(num_samples=100, num_frames=20, size=64, num_balls=6):
    """
    Generates bouncing balls dataset with specific number of balls.
    """
    
    # Precompute grid
    x = np.arange(size)
    y = np.arange(size)
    xx, yy = np.meshgrid(x, y)
    
    videos = []
    
    for _ in tqdm(range(num_samples), desc=f"Generating {num_balls}-ball videos"):
        # Initialize balls
        balls = []
        for _ in range(num_balls):
            pos = np.random.rand(2) * (size - 10) + 5
            vel = (np.random.rand(2) - 0.5) * 4 
            radius = 3 + np.random.rand() * 2
            color = np.random.rand(3)
            color = color / np.max(color)
            balls.append({'pos': pos, 'vel': vel, 'radius': radius, 'color': color})
            
        frames = []
        for t in range(num_frames):
            frame = np.zeros((size, size, 3))
            
            # Physics update
            for ball in balls:
                ball['pos'] += ball['vel']
                
                # Wall collisions
                if ball['pos'][0] < ball['radius']:
                    ball['pos'][0] = ball['radius']
                    ball['vel'][0] *= -1
                if ball['pos'][0] > size - ball['radius']:
                    ball['pos'][0] = size - ball['radius']
                    ball['vel'][0] *= -1
                if ball['pos'][1] < ball['radius']:
                    ball['pos'][1] = ball['radius']
                    ball['vel'][1] *= -1
                if ball['pos'][1] > size - ball['radius']:
                    ball['pos'][1] = size - ball['radius']
                    ball['vel'][1] *= -1
                    
                # Draw ball
                dist = np.sqrt((xx - ball['pos'][0])**2 + (yy - ball['pos'][1])**2)
                mask = dist <= ball['radius']
                frame[mask] = ball['color']
                
            frames.append(frame)
            
        videos.append(np.array(frames))
        
    videos = np.array(videos)
    # Transpose to (B, T, C, H, W)
    videos = videos.transpose(0, 1, 4, 2, 3)
    return torch.tensor(videos, dtype=torch.float32)

if __name__ == '__main__':
    os.makedirs('datasets/bouncing_balls_ood', exist_ok=True)
    
    # Generate 6-ball test set (Training was 3 balls)
    test_data = generate_bouncing_balls(num_samples=100, num_balls=6)
    torch.save(test_data, 'datasets/bouncing_balls_ood/test_6balls.pt')
    print("OOD Data saved to datasets/bouncing_balls_ood/test_6balls.pt")
