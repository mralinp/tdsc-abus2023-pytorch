from __future__ import annotations

import pandas as pd
from typing import Tuple, List, Callable, Optional
import numpy as np

from .tdsc import TDSC
from .enums import DataSplits


class TDSCTumors(TDSC):
    """
    A specialized class for the TDSC-ABUS2023 tumors dataset that handles
    bounding box information and tumor region extraction.
    """
    def __init__(
        self,
        path: str = "./data",
        split: DataSplits | str = DataSplits.TRAIN,
        transforms: Optional[List[Callable]] = None,
        download: bool = False
    ):
        """
        Initialize TDSCTumors dataset.

        Args:
            path: Base path for dataset storage
            split: Dataset split to use
            transforms: List of transformations to apply
            download: Whether to download the dataset if not found
        """
        super(TDSCTumors, self).__init__(path, split, transforms, download)

    def _extract_tumor_region(
        self,
        volume: np.ndarray,
        bbox_data: pd.Series
    ) -> Tuple[np.ndarray, Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
        """
        Extract tumor region from volume using bounding box coordinates.

        Args:
            volume: Input volume data
            bbox_data: Bounding box parameters

        Returns:
            Tuple of (extracted_volume, ((start_x, start_y, start_z), (end_x, end_y, end_z)))
        """
        # Calculate slice indices
        z_start = int(bbox_data['c_z'] - bbox_data['len_z']/2)
        z_end = int(bbox_data['c_z'] + bbox_data['len_z']/2)

        y_start = int(bbox_data['c_y'] - bbox_data['len_y']/2)
        y_end = int(bbox_data['c_y'] + bbox_data['len_y']/2)

        x_start = int(bbox_data['c_x'] - bbox_data['len_x']/2)
        x_end = int(bbox_data['c_x'] + bbox_data['len_x']/2)

        # Extract region (z, y, x order)
        extracted_volume = volume[z_start:z_end, y_start:y_end, x_start:x_end]
        bbox_coords = ((x_start, y_start, z_start), (x_end, y_end, z_end))

        return extracted_volume, bbox_coords

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Get a dataset item by index.

        Args:
            index: Index of the item to get

        Returns:
            Tuple containing (volume, mask, label), where volume and mask are
            cropped to the tumor region
        """
        # Load the untransformed volume/mask so the bounding box (which is defined
        # in the original coordinate space) crops the correct region. Transforms
        # are applied once, after cropping, instead of before AND after it.
        volume, mask, label, _ = self._get_raw_item(index)

        bbox_data = self.bbx_metadata.iloc[index]

        volume, _ = self._extract_tumor_region(volume, bbox_data)
        mask, _ = self._extract_tumor_region(mask, bbox_data)

        volume, mask = self._apply_transforms(volume, mask)

        label = 0 if label == 'M' else 1

        return volume, mask, label
