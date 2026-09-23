from __future__ import annotations

import os
import torch
import nrrd
import pandas as pd
from typing import Tuple, List, Callable, Optional, Any

from .enums import DataSplits
from .utils import DatasetDownloader


class TDSC(torch.utils.data.Dataset):
    """Base class for TDSC dataset handling volumetric medical data."""

    @property
    def class_names(self) -> dict[int, str]:
        return {
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
        self.split = split if isinstance(split, DataSplits) else DataSplits(split)
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
        metadata = pd.read_csv(
            os.path.join(self.path, str(self.split), "labels.csv"),
        )
        # Column casing varies across dataset releases; normalize once so the
        # rest of the class can rely on a single, predictable naming scheme.
        metadata.columns = [c.strip().lower() for c in metadata.columns]
        metadata['case_id'] = metadata['case_id'].astype(int)
        for col in ('label', 'data_path', 'mask_path'):
            metadata[col] = metadata[col].astype(str)
        self.metadata = metadata.set_index('case_id')

    def _load_bbox_metadata(self) -> None:
        """Load and prepare bounding box metadata."""
        bbox_path = os.path.join(self.path, str(self.split), "bbx_labels.csv")
        bbx_metadata = pd.read_csv(bbox_path)
        bbx_metadata.columns = [c.strip().lower() for c in bbx_metadata.columns]
        bbx_metadata['id'] = bbx_metadata['id'].astype(int)
        for col in ('c_x', 'c_y', 'c_z', 'len_x', 'len_y', 'len_z'):
            bbx_metadata[col] = bbx_metadata[col].astype(float)
        self.bbx_metadata = bbx_metadata.set_index('id')

        if len(self.bbx_metadata) != len(self.metadata):
            raise RuntimeError(
                f"labels.csv has {len(self.metadata)} rows but bbx_labels.csv has "
                f"{len(self.bbx_metadata)} rows for split '{self.split}'. The dataset "
                "files are out of sync."
            )

    def _check_exists(self) -> bool:
        """Check if the dataset files exist."""
        split_path = os.path.join(self.path, str(self.split))
        return (
            os.path.exists(os.path.join(split_path, "labels.csv"))
            and os.path.exists(os.path.join(split_path, "bbx_labels.csv"))
        )

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

    def _get_raw_item(
        self, index: int
    ) -> Tuple[Any, Any, str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """
        Load a dataset item by index without applying transforms.

        Returns:
            Tuple containing (volume, mask, raw_label, bbox), where raw_label is the
            untouched 'M'/'B' string and bbox is
            ((start_x, start_y, start_z), (end_x, end_y, end_z)) in the volume's
            original (untransformed) coordinate space.
        """
        row = self.metadata.iloc[index]
        label, vol_path, mask_path = row['label'], row['data_path'], row['mask_path']

        volume = self._load_volume(vol_path)
        mask = self._load_volume(mask_path)

        bbox_data = self.bbx_metadata.iloc[index]
        center = (bbox_data['c_x'], bbox_data['c_y'], bbox_data['c_z'])
        lengths = (bbox_data['len_x'], bbox_data['len_y'], bbox_data['len_z'])

        bbox = (
            tuple(c - l / 2 for c, l in zip(center, lengths)),  # start points
            tuple(c + l / 2 for c, l in zip(center, lengths))   # end points
        )

        return volume, mask, label, bbox

    def __getitem__(self, index: int) -> Tuple[Any, Any, int, Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """
        Get a dataset item by index.

        Args:
            index: Index of the item to get

        Returns:
            Tuple containing (volume, mask, label, bbox), where bbox is ((start_x, start_y, start_z), (end_x, end_y, end_z))
        """
        volume, mask, label, bbox = self._get_raw_item(index)
        volume, mask = self._apply_transforms(volume, mask)
        label = 0 if label == 'M' else 1
        return volume, mask, label, bbox

    def __len__(self) -> int:
        """Get the total number of items in the dataset."""
        return len(self.metadata)
