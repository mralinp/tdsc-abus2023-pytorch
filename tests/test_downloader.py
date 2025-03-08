import pytest
import json
import os
from unittest.mock import patch, MagicMock
from pathlib import Path
from tdsc_abus2023_pytorch.utils.downloader import DatasetDownloader
from tdsc_abus2023_pytorch import DataSplits

@pytest.fixture
def mock_json_response():
    """Mock response for the GitHub raw file."""
    return {
        "Train": "1i0nLWX2PXiJnE_pZd79XyV1zshETCgWz",
        "Validation": "1--ldK9wb-DGKuKPO0ro-Ka6pRtndIekh",
        "Test": "1z7gGTR2w45VySccvstZkgycnQx3qeTVa"
    }

@pytest.fixture
def mock_requests_get(mock_json_response):
    """Mock requests.get for GitHub raw file."""
    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = mock_json_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        yield mock_get

@pytest.fixture
def mock_gdown():
    """Mock gdown.download."""
    with patch('gdown.download') as mock_download:
        mock_download.return_value = True
        yield mock_download

def test_get_file_ids(mock_requests_get, mock_json_response):
    """Test fetching file IDs from GitHub."""
    file_ids = DatasetDownloader.get_file_ids()
    assert file_ids == mock_json_response
    mock_requests_get.assert_called_once_with(DatasetDownloader.GITHUB_RAW_URL)

def test_download_dataset(mock_requests_get, mock_gdown, tmp_path):
    """Test downloading a specific split."""
    DatasetDownloader.download_dataset(DataSplits.TRAIN, base_path=str(tmp_path))
    
    # Check if directory was created
    assert (tmp_path / "tdsc" / "Train").exists()
    
    # Verify gdown was called with correct ID
    mock_gdown.assert_called_once_with(
        f"https://drive.google.com/uc?id=1i0nLWX2PXiJnE_pZd79XyV1zshETCgWz",
        str(tmp_path / "tdsc" / "Train.zip"),
        quiet=False
    )

def test_download_dataset_skips_existing(mock_requests_get, mock_gdown, tmp_path):
    """Test if download_dataset skips existing files."""
    # Create mock existing file
    data_dir = tmp_path / "tdsc" / "Train"
    data_dir.mkdir(parents=True)
    (data_dir / "data.nrrd").touch()

    DatasetDownloader.download_dataset(DataSplits.TRAIN, base_path=str(tmp_path))
    assert mock_gdown.call_count == 0

def test_download_all_splits(mock_requests_get, mock_gdown, tmp_path):
    """Test downloading all splits."""
    DatasetDownloader.download_all(base_path=str(tmp_path))
    
    # Should call download for each split
    assert mock_gdown.call_count == len(DataSplits)
    
    # Verify all directories exist
    for split in DataSplits:
        assert (tmp_path / "tdsc" / str(split)).exists()

def test_invalid_split(mock_requests_get, mock_gdown, tmp_path):
    """Test handling of invalid split."""
    with pytest.raises(ValueError, match="Invalid split"):
        DatasetDownloader.download_dataset("InvalidSplit", base_path=str(tmp_path))

def test_failed_github_request(mock_requests_get, tmp_path):
    """Test handling of failed GitHub request."""
    mock_requests_get.side_effect = Exception("Failed to fetch")
    
    with pytest.raises(RuntimeError, match="Failed to fetch file IDs"):
        DatasetDownloader.download_dataset(DataSplits.TRAIN, base_path=str(tmp_path))

def test_failed_download(mock_requests_get, mock_gdown, tmp_path):
    """Test handling of failed download."""
    mock_gdown.return_value = None  # Simulate failed download
    
    with pytest.raises(RuntimeError, match="Failed to download"):
        DatasetDownloader.download_dataset(DataSplits.TRAIN, base_path=str(tmp_path)) 