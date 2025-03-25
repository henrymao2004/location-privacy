import tensorflow as tf
import numpy as np

class TrajectoryDataset:
    def __init__(self, data, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.num_samples = len(data)
        self.num_batches = (self.num_samples + batch_size - 1) // batch_size
        
    def __len__(self):
        return self.num_batches
    
    def __iter__(self):
        self.current_batch = 0
        return self
    
    def __next__(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration
            
        start_idx = self.current_batch * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)
        
        # Get batch of trajectories
        batch_data = self.data[start_idx:end_idx]
        
        # Prepare batch data
        batch = {}
        for key in batch_data[0].keys():
            batch[key] = np.stack([traj[key] for traj in batch_data])
        
        self.current_batch += 1
        return batch
    
    def get_batch(self, batch_idx):
        """Get a specific batch by index"""
        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)
        
        batch_data = self.data[start_idx:end_idx]
        batch = {}
        for key in batch_data[0].keys():
            batch[key] = np.stack([traj[key] for traj in batch_data])
        
        return batch
    
    def shuffle(self):
        """Shuffle the dataset"""
        np.random.shuffle(self.data)
    
    def get_all_data(self):
        """Get all data as a single batch"""
        batch = {}
        for key in self.data[0].keys():
            batch[key] = np.stack([traj[key] for traj in self.data])
        return batch 