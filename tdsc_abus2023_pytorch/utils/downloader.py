import json
import os
from importlib import resources
from zipfile import ZipFile

import gdown


class DatasetDownloader:

    @classmethod
    def get_file_ids(cls) -> dict:
        """Load the Google Drive file ID for each dataset split, bundled with the package."""
        with resources.files("tdsc_abus2023_pytorch.resources").joinpath("gdrive_files.json").open("r") as f:
            return json.load(f)

    @classmethod
    def download_dataset(cls, split: str, base_path=None) -> None:
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
            gdrive_id = file_ids[str(split)]
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
    def download_all(cls, base_path=None) -> None:
        """Download all dataset splits."""
        from .. import DataSplits
        for split in DataSplits:
            cls.download_dataset(split, base_path)
