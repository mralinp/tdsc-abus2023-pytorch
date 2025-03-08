import os
import json
import requests
from pathlib import Path
import gdown

class DatasetDownloader:
    GITHUB_RAW_URL = "https://raw.githubusercontent.com/mralinp/tdsc-abus2023-pytorch/main/tdsc-abus2023-pytorch/resources/gdrive_files.json"

    @classmethod
    def get_file_ids(cls):
        """Fetch the file IDs from GitHub."""
        response = requests.get(cls.GITHUB_RAW_URL)
        response.raise_for_status()
        return response.json()

    @classmethod
    def download_dataset(cls, split, base_path=None):
        """
        Download dataset files for a specific split.
        
        Args:
            split (DataSplits): The dataset split to download
            base_path (str, optional): Base path to store the dataset. Defaults to ./data
        """
        if base_path is None:
            base_path = os.path.join(os.getcwd(), "data")

        file_ids = cls.get_file_ids()
        split_data = file_ids["tdsc"][split]

        for data_type, info in split_data.items():
            output_path = os.path.join(base_path, info["path"])
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            if not os.path.exists(output_path):
                print(f"Downloading {split} {data_type}...")
                url = f"https://drive.google.com/uc?id={info['gdrive_id']}"
                gdown.download(url, output_path, quiet=False)
            else:
                print(f"File already exists: {output_path}")

    @classmethod
    def download_all(cls, base_path=None):
        """Download all dataset splits."""
        from .. import DataSplits
        for split in DataSplits:
            cls.download_dataset(split, base_path) 