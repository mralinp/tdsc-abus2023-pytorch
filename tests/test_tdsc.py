import unittest
import torch
from tdsc_abus2023_pytorch import TDSC

class TestTDSCDataset(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.dataset = TDSC(path="./data/tdsc", split="Train")

    def test_dataset_length(self):
        """Test if the dataset has the correct length."""
        self.assertGreater(len(self.dataset), 0)

    def test_dataset_getitem(self):
        """Test if the dataset returns correct tensor types and shapes."""
        volume, mask, label = self.dataset[0]
        
        # Check types
        self.assertIsInstance(volume, torch.Tensor)
        self.assertIsInstance(mask, torch.Tensor)
        self.assertIsInstance(label, torch.Tensor)
        
        # Check dimensions (adjust expected shapes based on your dataset)
        self.assertEqual(len(volume.shape), 3)  # 3D volume
        self.assertEqual(len(mask.shape), 3)    # 3D mask
        self.assertEqual(len(label.shape), 1)   # 1D label

if __name__ == '__main__':
    unittest.main() 