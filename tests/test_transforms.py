import unittest
import numpy as np
import torch
from tdsc import TDSC

class MockTransform:
    def __call__(self, volume, mask):
        # Simple transform that adds 1 to volume and mask
        return volume + 1, mask + 1

class TestTDSCTransforms(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.transforms = [MockTransform()]
        self.dataset = TDSC(
            path="./data/tdsc",
            split="Train",
            transforms=self.transforms
        )

    def test_transform_application(self):
        """Test if transforms are properly applied to volume and mask."""
        volume, mask, label = self.dataset[0]
        
        # Since our MockTransform adds 1, all values should be increased
        self.assertTrue(torch.all(volume > 0))  # Assuming original data has zeros
        self.assertTrue(torch.all(mask > 0))    # Assuming original mask has zeros

    def test_multiple_transforms(self):
        """Test if multiple transforms are applied in sequence."""
        # Create dataset with multiple transforms
        multi_transforms = [MockTransform(), MockTransform()]
        dataset = TDSC(
            path="./data/tdsc",
            split="Train",
            transforms=multi_transforms
        )
        
        volume, mask, label = dataset[0]
        
        # Since we applied MockTransform twice, values should be increased by 2
        self.assertTrue(torch.all(volume > 1))
        self.assertTrue(torch.all(mask > 1))

    def test_invalid_transform(self):
        """Test if non-callable transform raises error."""
        with self.assertRaises(TypeError):
            TDSC(
                path="./data/tdsc",
                split="Train",
                transforms=[lambda x, y: (x, y), "not_a_callable"]
            )

if __name__ == '__main__':
    unittest.main() 