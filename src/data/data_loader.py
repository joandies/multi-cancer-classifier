import os
from torch.utils.data import DataLoader
from torchvision import transforms
from cervical_dataset import CervicalCancerDataset

def get_data_loaders(data_dir, batch_size, num_workers=4):
    """
    Creates DataLoaders for train, validation, and test datasets.
    
    Args:
        data_dir (str): Path to the root directory of the dataset.
        batch_size (int): Batch size for training and evaluation.
        num_workers (int): Number of worker threads for data loading.

    Returns:
        dict: A dictionary containing DataLoaders for 'train', 'val', 'test'.
    """
    # Define transformations for the dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize images to [-1, 1]
    ])

    data_loaders = {}
    for split in ['train', 'val', 'test']:
        dataset = CervicalCancerDataset(root_dir=os.path.join(data_dir, split), transform=transform)
        data_loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),  # Shuffle only training data
            num_workers=num_workers
        )
    return data_loaders