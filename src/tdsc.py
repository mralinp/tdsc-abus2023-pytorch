import os
import torch
import nrrd
import pandas as pd
from typing import Tuple, List, Callable, Optional, Any

from .enums import DataSplits
from .utils.downloader import DatasetDownloader

class TDSC(torch.utils.data.Dataset):
    """Base class for TDSC dataset handling volumetric medical data."""

    class_names = {
        0: 'Malignant',  # 'M' in the dataset
        1: 'Benign'      # 'B' in the dataset
    }

    def __init__(
        self, 
        path: str = "./data",
        split: DataSplits | str = DataSplits.TRAIN,
        transforms: Optional[List[Callable]] = None,
        download: bool = False
    ):
        """
        Initialize TDSC dataset.

        Args:
            path: Base path for dataset storage
            split: Dataset split to use
            transforms: List of transformations to apply
            download: Whether to download the dataset if not found
        """
        self.path = path
        self.split = DataSplits(split) if isinstance(split, str) else split
        self.transforms = transforms or []
        self.do_transform = True

        if download:
            self._download_if_needed()

        if not self._check_exists():
            raise RuntimeError(
                f"Dataset not found in {self.path}. "
                "You can use download=True to download it"
            )
        self._load_metadata()
        self._load_bbox_metadata()

    def _load_metadata(self) -> None:
        """Load and prepare dataset metadata."""
        self.metadata = pd.read_csv(
            os.path.join(self.path, str(self.split), "labels.csv"),
            dtype={
                'Case_id': int,
                'Label': str,
                'Data_path': str,
                'Mask_path': str
            }
        ).set_index('case_id')
        
    def _load_bbox_metadata(self) -> None:
        """Load and prepare bounding box metadata."""
        bbox_path = os.path.join(self.path, str(self.split), "bbx_labels.csv")
        self.bbx_metadata = pd.read_csv(
            bbox_path,
            dtype={
                'id': int,
                'c_x': float,
                'c_y': float,
                'c_z': float,
                'len_x': float,
                'len_y': float,
                'len_z': float,
            },
            index_col='id'
        )

    def _check_exists(self) -> bool:
        """Check if the dataset files exist."""
        split_path = os.path.join(self.path, str(self.split))
        return os.path.exists(split_path)

    def _download_if_needed(self) -> None:
        """Download dataset if it doesn't exist."""
        DatasetDownloader.download_dataset(self.split, self.path)

    def _load_volume(self, path: str) -> Any:
        """Load a volume file from disk."""
        full_path = os.path.join(self.path, str(self.split), path.replace('\\', '/'))
        volume, _ = nrrd.read(full_path)
        return volume

    def _apply_transforms(self, volume: Any, mask: Any) -> Tuple[Any, Any]:
        """Apply transformations to volume and mask."""
        if self.transforms and self.do_transform:
            for transformer in self.transforms:
                volume, mask = transformer(volume, mask)
        return volume, mask

    def __getitem__(self, index: int) -> Tuple[Any, Any, int, Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """
        Get a dataset item by index.

        Args:
            index: Index of the item to get

        Returns:
            Tuple containing (volume, mask, label, bbox), where bbox is ((start_x, start_y, start_z), (end_x, end_y, end_z))
        """
        label, vol_path, mask_path = self.metadata.iloc[index]
        
        # Load volume and mask
        volume = self._load_volume(vol_path)
        mask = self._load_volume(mask_path)
        
        # Get bounding box data
        bbox_data = self.bbx_metadata.iloc[index]
        center = (bbox_data['c_x'], bbox_data['c_y'], bbox_data['c_z'])
        lengths = (bbox_data['len_x'], bbox_data['len_y'], bbox_data['len_z'])
        
        # Calculate start and end points
        bbox = (
            tuple(c - l/2 for c, l in zip(center, lengths)),  # start points
            tuple(c + l/2 for c, l in zip(center, lengths))   # end points
        )
        
        # Apply transformations
        volume, mask = self._apply_transforms(volume, mask)
        
        # Convert label
        label = 0 if label == 'M' else 1
        
        return volume, mask, label, bbox

    def __len__(self) -> int:
        """Get the total number of items in the dataset."""
        return len(self.metadata)