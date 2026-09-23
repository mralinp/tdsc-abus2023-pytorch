<h1 align="center">TDSC-ABUS2023 · PyTorch</h1>

<p align="center">
  <b>A lightweight PyTorch <code>Dataset</code> for the TDSC-ABUS 2023 challenge</b><br>
  Tumor Detection, Segmentation and Classification on Automated 3D Breast Ultrasound
</p>

<p align="center">
  <a href="https://pypi.org/project/tdsc-abus2023-pytorch/"><img src="https://img.shields.io/pypi/v/tdsc-abus2023-pytorch?color=2b6cb0" alt="PyPI"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.9%2B-2b6cb0" alt="Python 3.9+"></a>
  <a href="https://arxiv.org/abs/2501.15588"><img src="https://img.shields.io/badge/arXiv-2501.15588-b31b1b" alt="arXiv"></a>
  <a href="https://github.com/mralinp/tdsc-abus2023-pytorch/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-2b6cb0" alt="MIT License"></a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2501.15588">Paper</a> ·
  <a href="https://tdsc-abus2023.grand-challenge.org/">Challenge</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/mralinp/tdsc-abus2023-pytorch/main/assets/sample_case.png" width="420" alt="Axial slice with tumor mask and bounding box, next to the TDSCTumors crop">
</p>
<p align="center">
  <sub><b>Figure 1.</b> Case 8 (malignant). <b>(a)</b> Axial slice of the full volume returned by <code>TDSC</code>, with the tumor mask and bounding box. <b>(b)</b> The same tumor as returned by <code>TDSCTumors</code>, cropped to its bounding box.</sub>
</p>

---

## Highlights

- **Zero setup.** Each split is downloaded and extracted on first use, then cached locally.
- **Ready-to-train samples.** Volumes, masks, labels and tumor bounding boxes, straight from `dataset[i]`.
- **Tumor crops.** `TDSCTumors` yields volumes already cropped to the tumor, for classification or patch-based segmentation.
- **Minimal dependencies.** `torch`, `numpy`, `pandas`, `pynrrd` and `gdown`, nothing else.

## Installation

```bash
pip install tdsc-abus2023-pytorch
```

Requires Python 3.9 or newer.

## Quick Start

### Full volumes

```python
from tdsc_abus2023_pytorch import TDSC, DataSplits

dataset = TDSC(path="./data", split=DataSplits.TRAIN, download=True)
volume, mask, label, bbox = dataset[0]
```

| Output   | Type                               | Description                                          |
| -------- | ---------------------------------- | ---------------------------------------------------- |
| `volume` | `np.ndarray`                       | Full 3D ultrasound volume                            |
| `mask`   | `np.ndarray`                       | Segmentation mask (`0` background, `1` tumor)        |
| `label`  | `int`                              | `0` malignant, `1` benign                            |
| `bbox`   | `((x0, y0, z0), (x1, y1, z1))`     | Tumor bounding box in the volume's native coordinates |

### Tumor crops

```python
from tdsc_abus2023_pytorch import TDSCTumors, DataSplits

dataset = TDSCTumors(path="./data", split=DataSplits.TRAIN, download=True)
volume, mask, label = dataset[0]
```

### Anatomical views

`ViewTransformer` transposes each volume/mask pair into the requested anatomical plane.

```python
from tdsc_abus2023_pytorch import TDSC, DataSplits, ViewTransformer, ViewTransposeConfig

dataset = TDSC(
    path="./data",
    split=DataSplits.TRAIN,
    transforms=[ViewTransformer(view=ViewTransposeConfig.CORONAL)],
)
```

<p align="center">
  <img src="https://raw.githubusercontent.com/mralinp/tdsc-abus2023-pytorch/main/assets/views.png" width="100%" alt="Axial, coronal and sagittal views of the same tumor produced by ViewTransformer">
</p>
<p align="center">
  <sub><b>Figure 2.</b> Slice through the tumor of case 8 in each view produced by <code>ViewTransformer</code>: <b>(a)</b> axial, <b>(b)</b> coronal, <b>(c)</b> sagittal.</sub>
</p>

### Custom transforms

Any callable `(volume, mask) -> (volume, mask)` works as a transform. Transforms run in the order given.

```python
class MyTransform:
    def __call__(self, volume, mask):
        return volume, mask  # your logic here

dataset = TDSC(
    path="./data",
    split=DataSplits.TRAIN,
    transforms=[ViewTransformer(ViewTransposeConfig.AXIAL), MyTransform()],
)
```

## Dataset

The dataset has 200 3D breast ultrasound volumes acquired with an Invenia ABUS system (GE Healthcare) at Harbin Medical University Cancer Hospital, China. Experienced radiologists annotated and verified the tumor masks and bounding boxes.

| Split      | Cases | Malignant | Benign |
| ---------- | ----: | --------: | -----: |
| Train      |   100 |        58 |     42 |
| Validation |    30 |        17 |     13 |
| Test       |    70 |        40 |     30 |
| **Total**  | **200** | **115** | **85** |

| Property      | Value                                            |
| ------------- | ------------------------------------------------ |
| Volume size   | 843×546×270 to 865×682×354 voxels                |
| Voxel spacing | 0.200 × 0.073 mm (X–Y), ~0.476 mm (Z)            |
| File format   | `.nrrd`                                          |

With `download=True`, each requested split is fetched from Google Drive, extracted and cached under `path/<Split>/`. Later runs reuse the cached copy and don't touch the network:

```
data/
├── Train/
│   ├── DATA/            # volumes (.nrrd)
│   ├── MASK/            # masks (.nrrd)
│   ├── labels.csv
│   └── bbx_labels.csv
├── Validation/
└── Test/
```

## API Reference

| Class                                              | Returns                        | Notes                                                      |
| -------------------------------------------------- | ------------------------------ | ---------------------------------------------------------- |
| `TDSC(path, split, transforms, download)`          | `(volume, mask, label, bbox)`  | Full volume and mask                                       |
| `TDSCTumors(path, split, transforms, download)`    | `(volume, mask, label)`        | Cropped to the tumor bounding box                          |
| `ViewTransformer(view)`                            | `(volume, mask)`               | `ViewTransposeConfig.{AXIAL, CORONAL, SAGITTAL}`           |
| `DataSplits`                                       | —                              | `DataSplits.{TRAIN, VALIDATION, TEST}`                     |

`split` also accepts a plain string, for example `split="Train"`.

## Development

```bash
git clone https://github.com/mralinp/tdsc-abus2023-pytorch.git
cd tdsc-abus2023-pytorch
pip install -r requirements.txt pytest pytest-cov
pytest
```

The test suite runs offline against a small synthetic dataset and never downloads the real data. Contributions are welcome through pull requests.

## Citation

If you use this dataset, please cite the challenge paper:

```bibtex
@misc{luo2025tumordetectionsegmentationclassification,
  title         = {Tumor Detection, Segmentation and Classification Challenge on Automated 3D Breast Ultrasound: The TDSC-ABUS Challenge},
  author        = {Gongning Luo and others},
  year          = {2025},
  eprint        = {2501.15588},
  archivePrefix = {arXiv},
  primaryClass  = {eess.IV},
  url           = {https://arxiv.org/abs/2501.15588}
}
```

## License

The code is released under the [MIT License](https://github.com/mralinp/tdsc-abus2023-pytorch/blob/main/LICENSE). The dataset itself is provided by the TDSC-ABUS 2023 organizers and is subject to their terms of use.
