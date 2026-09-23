import numpy as np
import pytest

from tdsc_abus2023_pytorch import DataSplits, TDSC, TDSCTumors, ViewTransformer, ViewTransposeConfig


def test_tdsc_instantiation(dataset_dir):
    dataset = TDSC(path=dataset_dir, split=DataSplits.TRAIN)
    assert len(dataset) == 1
    assert dataset.path == dataset_dir
    assert dataset.split == DataSplits.TRAIN
    assert len(dataset.transforms) == 0


def test_tdsc_getitem(dataset_dir, volume_data):
    volume, mask = volume_data
    dataset = TDSC(path=dataset_dir, split=DataSplits.TRAIN)

    vol, msk, label, bbox = dataset[0]

    np.testing.assert_array_equal(vol, volume)
    np.testing.assert_array_equal(msk, mask)
    assert label == 0  # 'M' -> Malignant
    assert bbox == ((1.0, 1.0, 2.0), (3.0, 3.0, 4.0))


def test_tdsc_missing_dataset_raises_without_download(tmp_path):
    with pytest.raises(RuntimeError):
        TDSC(path=str(tmp_path), split=DataSplits.TRAIN, download=False)


def test_tdsc_tumors_crops_to_bounding_box(dataset_dir, volume_data):
    volume, mask = volume_data
    expected_volume_crop = volume[2:4, 1:3, 1:3]
    expected_mask_crop = mask[2:4, 1:3, 1:3]

    dataset = TDSCTumors(path=dataset_dir, split=DataSplits.TRAIN)
    cropped_volume, cropped_mask, label = dataset[0]

    np.testing.assert_array_equal(cropped_volume, expected_volume_crop)
    np.testing.assert_array_equal(cropped_mask, expected_mask_crop)
    assert label == 0


def test_tdsc_tumors_crops_before_applying_transform(dataset_dir, volume_data):
    """
    Regression test: TDSCTumors must crop the tumor region using the bounding
    box (defined in the volume's original coordinate space) BEFORE applying any
    view transform, and must apply each transform exactly once. Applying the
    transform first (or twice) shifts which voxels the bounding box selects.
    """
    volume, mask = volume_data
    expected_crop = volume[2:4, 1:3, 1:3]
    expected = np.transpose(expected_crop, (1, 2, 0))

    transformer = ViewTransformer(view=ViewTransposeConfig.CORONAL)
    dataset = TDSCTumors(path=dataset_dir, split=DataSplits.TRAIN, transforms=[transformer])
    cropped_volume, _, _ = dataset[0]

    np.testing.assert_array_equal(cropped_volume, expected)


@pytest.mark.parametrize("cls", [TDSC, TDSCTumors])
def test_cache_matches_uncached_and_writes_npy(dataset_dir, cls):
    import os
    expected = cls(path=dataset_dir)[0]
    cached = cls(path=dataset_dir, cache=True)
    for _ in range(2):  # first call builds the .npy, second reads it via mmap
        got = cached[0]
        for e, g in zip(expected[:2], got[:2]):
            np.testing.assert_array_equal(g, e)
            assert type(g) is np.ndarray and g.flags.writeable
        assert got[2:] == expected[2:]
    assert os.path.exists(os.path.join(dataset_dir, "Train", "DATA", "case1.npy"))
