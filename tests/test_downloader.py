import pytest
from unittest.mock import patch, MagicMock
from tdsc_abus2023_pytorch.utils.downloader import DatasetDownloader
from tdsc_abus2023_pytorch import DataSplits
import os


def test_get_file_ids():
    """Test fetching file IDs from GitHub."""
    file_ids = DatasetDownloader.get_file_ids()
    assert file_ids is not None
    
def test_download_dataset():
    DatasetDownloader.download_dataset(DataSplits.TEST)
    assert os.path.exists("data/Test")
    
def test_download_all():
    DatasetDownloader.download_all()
    assert os.path.exists("data/Train")
    assert os.path.exists("data/Validation")
    assert os.path.exists("data/Test")
