
import numpy as np
import torch
import os
import argparse
from tqdm import tqdm

def generate_bouncing_balls(num_samples=1000, num_frames=20, size=64, num_balls=3):
    """
    Generates bouncing balls dataset.
    Returns:
        video: (num_samples, num_frames, 3, size, size)
    """
    
    # Precompute grid
    x = np.arange(size)
    y = np.arange(size)
    xx, yy = np.meshgrid(x, y)
    
    videos = []
    
    for _ in tqdm(range(num_samples), desc="Generating videos"):
        # Initialize balls
        balls = []
        for _ in range(num_balls):
            pos = np.random.rand(2) * (size - 10) + 5
            vel = (np.random.rand(2) - 0.5) * 4 # Random velocity
            radius = 3 + np.random.rand() * 2
            color = np.random.rand(3)
            # Ensure bright colors
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_samples', type=int, default=1000)
    parser.add_argument('--test_samples', type=int, default=100)
    parser.add_argument('--num_frames', type=int, default=20)
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--out_dir', type=str, default='datasets/bouncing_balls')
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("Generating Training Data...")
    train_data = generate_bouncing_balls(args.train_samples, args.num_frames, args.size)
    torch.save(train_data, os.path.join(args.out_dir, 'train.pt'))
    
    print("Generating Test Data...")
    test_data = generate_bouncing_balls(args.test_samples, args.num_frames, args.size)
    torch.save(test_data, os.path.join(args.out_dir, 'test.pt'))
    
    print(f"Data saved to {args.out_dir}")
    print(f"Train shape: {train_data.shape}")
    print(f"Test shape: {test_data.shape}")

if __name__ == '__main__':
    main()
