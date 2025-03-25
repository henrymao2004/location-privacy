import torch
import numpy as np
from torch.utils.data import Dataset

class TrajectoryDataset(Dataset):
    def __init__(self, data, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.num_samples = len(data)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Get trajectory data
        traj_data = self.data[idx]
        
        # Convert numpy arrays to PyTorch tensors
        batch = {}
        for key in traj_data.keys():
            batch[key] = torch.from_numpy(traj_data[key]).float()
        
        return batch
    
    def get_batch(self, batch_idx):
        """Get a specific batch by index"""
        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)
        
        batch_data = self.data[start_idx:end_idx]
        batch = {}
        for key in batch_data[0].keys():
            batch[key] = torch.stack([torch.from_numpy(traj[key]).float() 
                                    for traj in batch_data])
        
        return batch
    
    def get_all_data(self):
        """Get all data as a single batch"""
        batch = {}
        for key in self.data[0].keys():
            batch[key] = torch.stack([torch.from_numpy(traj[key]).float() 
                                    for traj in self.data])
        return batch 