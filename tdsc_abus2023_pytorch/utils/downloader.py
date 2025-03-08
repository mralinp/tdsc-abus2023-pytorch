import os
import json
import requests
from pathlib import Path
import gdown
from zipfile import ZipFile

class DatasetDownloader:
    GITHUB_RAW_URL = "https://raw.githubusercontent.com/mralinp/tdsc-abus2023-pytorch/main/tdsc_abus2023_pytorch/resources/gdrive_files.json"

    @classmethod
    def get_file_ids(cls):
        """Fetch the file IDs from GitHub."""
        response = requests.get(cls.GITHUB_RAW_URL)
        response.raise_for_status()
        return response.json()

    @classmethod
    def download_dataset(cls, split: str, base_path=None):
        """
        Download dataset files for a specific split, unzip them, and remove the zip file.
        
        Args:
            split (DataSplits): The dataset split to download
            base_path (str, optional): Base path to store the dataset. Defaults to ./data
        """
        if base_path is None:
            base_path = os.path.join(os.getcwd(), "data")

        # Construct the output paths
        output_filename = f"{split}.zip"
        output_path = os.path.join(base_path, output_filename)
        split_path = os.path.join(base_path, str(split))
        
        # If the dataset folder already exists, skip everything
        if os.path.exists(split_path):
            print(f"Dataset already exists in: {split_path}")
            return

        os.makedirs(base_path, exist_ok=True)

        # If zip doesn't exist, download it
        if not os.path.exists(output_path):
            print(f"Downloading {split} dataset...")
            file_ids = cls.get_file_ids()
            gdrive_id = file_ids[split]
            url = f"https://drive.google.com/uc?id={gdrive_id}"
            gdown.download(url, output_path, quiet=False)
        else:
            print(f"Zip file already exists: {output_path}")
        
        print(f"Extracting {output_filename}...")
        with ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall(base_path)
        
        print(f"Removing {output_filename}...")
        os.remove(output_path)

    @classmethod
    def download_all(cls, base_path=None):
        """Download all dataset splits."""
        from .. import DataSplits
        for split in DataSplits:
            cls.download_dataset(split, base_path) 