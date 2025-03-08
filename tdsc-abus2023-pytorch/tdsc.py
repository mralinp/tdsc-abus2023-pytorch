import os
import nrrd
import pandas as pd
from typing import Final

from dataset import Dataset
from google_drive_downloader import GoogleDriveDownloader
from downloader import DatasetDownloader
from __init__ import DataSplits


DATASET_IDS: Final = {
    "train":{
    
    "data_0.zip": {
        "id":"1K762mw8vAIRjgoeR7aJN-0NIntOBUb6O",
        "path": "DATA",
        "zip": True
    },
    "data_1.zip": {
        "id":"17Umzl10lpFu4mGJ9HrZrC-Lcc-klKcl1",
        "path": "DATA",
        "zip": True
    },
    "mask.zip": {
        "id":"1Z2RUoUoOukA93LyTgwKPKCrfFePva1pV",
        "path": "MASK",
        "zip": True
    },
    "labels.csv": {
        "id":"1Fn6psOjknovxmShESRYpaxDAbb9txvH7",
        "path": "."
    },
    "bbx_labels.csv": {
        "id":"1firgUGMMMscXoYlQCzdg8Y7x2Hc3enIt",
        "path": "."
    }
    },
    "validation":{},
    "test":{}
}


class TDSC(Dataset):

    def __init__(self, path="./data", split=DataSplits.TRAIN, transforms=None, download=False):
        """
        Args:
            path (str): Path to the dataset
            split (DataSplits): Which split to use
            transforms (list): List of callable transforms
            download (bool): Whether to download the dataset if not found
        """
        self.path = path
        self.split = DataSplits(split) if isinstance(split, str) else split
        self.transforms = transforms or []

        if download:
            DatasetDownloader.download_dataset(self.split, self.path)

        if not self._check_exists():
            raise RuntimeError(
                f"Dataset not found in {self.path}. "
                "You can use download=True to download it"
            )
        
        super(TDSC, self).__init__(path)
        self.do_transform = True
        self.metadata = pd.read_csv(f"{self.path}/{self.split}/labels.csv", 
                                  dtype={'Case_id': int, 'Label': str, 'Data_path': str, 'Mask_path': str}).set_index('case_id')
        
    def _check_exists(self):
        """Check if the dataset files exist."""
        split_path = os.path.join(self.path, "tdsc", str(self.split))
        return os.path.exists(os.path.join(split_path, "DATA")) and \
               os.path.exists(os.path.join(split_path, "MASK"))
        
    def validate(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            os.makedirs(f"{self.path}/{self.split}/DATA")
            os.makedirs(f"{self.path}/{self.split}/MASK")
        self.download_data()
        return True
    
    def download_data(self):
        downloader = GoogleDriveDownloader(None)
        for file_name, file_info in DATASET_IDS.items():
            downloader.download(file_info.get("id"), f"{self.path}/{self.split}/{file_info.get('path')}/{file_name}")
            if file_info.get("zip"):
                downloader.save()
        
    def __getitem__(self, index) -> tuple:
        label, vol_path, mask_path = self.metadata.iloc[index]
        vol_path = vol_path.replace('\\', '/')
        mask_path = mask_path.replace('\\', '/')
        label = 0 if label == 'M' else 1
        
        vol, _ = nrrd.read(f"{self.path}/{self.split}/{vol_path}")
        mask, _ = nrrd.read(f"{self.path}/{self.split}/{mask_path}") 
        
        if self.transforms and self.do_transform:
            for transformer in self.transforms:
                vol, mask = transformer(vol, mask)
        
        return vol, mask, label

    def __len__(self) -> int:
        return len(self.metadata)