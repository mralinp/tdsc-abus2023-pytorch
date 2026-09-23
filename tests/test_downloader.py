from unittest.mock import patch
from zipfile import ZipFile

from tdsc_abus2023_pytorch import DataSplits
from tdsc_abus2023_pytorch.utils import DatasetDownloader


def test_get_file_ids_reads_bundled_resource():
    """File IDs are read from the JSON bundled with the package, no network access."""
    file_ids = DatasetDownloader.get_file_ids()
    assert set(file_ids) == {"Train", "Validation", "Test"}


def test_download_dataset_skips_when_already_present(tmp_path):
    (tmp_path / "Test").mkdir()

    with patch("tdsc_abus2023_pytorch.utils.downloader.gdown.download") as mock_download:
        DatasetDownloader.download_dataset(DataSplits.TEST, str(tmp_path))

    mock_download.assert_not_called()


def test_download_dataset_downloads_extracts_and_cleans_up(tmp_path):
    def fake_download(url, output_path, quiet=False):
        with ZipFile(output_path, "w") as zf:
            zf.writestr("Test/labels.csv", "Case_id,Label,Data_path,Mask_path\n")
        return output_path

    with patch(
        "tdsc_abus2023_pytorch.utils.downloader.gdown.download",
        side_effect=fake_download,
    ) as mock_download:
        DatasetDownloader.download_dataset(DataSplits.TEST, str(tmp_path))

    mock_download.assert_called_once()
    assert (tmp_path / "Test" / "labels.csv").exists()
    assert not (tmp_path / "Test.zip").exists()
