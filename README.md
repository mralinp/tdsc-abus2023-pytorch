# TDSC-ABUS2023 PyTorch Dataset

[![PyPI version](https://img.shields.io/pypi/v/tdsc-abus2023-pytorch)](https://pypi.org/project/tdsc-abus2023-pytorch/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/mralinp/tdsc-abus2023-pytorch/python-publish.yml?branch=main)](https://github.com/mralinp/tdsc-abus2023-pytorch/actions)

A PyTorch-compatible dataset package containing volumetric data from the **TDSC-ABUS2023** collection (**Tumor Detection, Segmentation, and Classification Challenge on Automated 3D Breast Ultrasound**).

---

## 📊 Dataset Description

The dataset consists of **200 3D ultrasound volumes** collected using an **Invenia ABUS (GE Healthcare)** system at **Harbin Medical University Cancer Hospital, China**. All tumor annotations were created and verified by experienced radiologists.

### Dataset Composition

| **Set**       | **Cases** | **Malignant** | **Benign** |
|--------------|----------|--------------|------------|
| **Training**  | 100      | 58           | 42         |
| **Validation**| 30       | 17           | 13         |
| **Test**      | 70       | 40           | 30         |

### Technical Specifications
- **Image Dimensions**: Vary between **843×546×270** and **865×682×354**  
- **Pixel Spacing**:
  - X-Y plane: **0.200 mm × 0.073 mm**
  - Z-axis (between slices): **~0.475674 mm**
- **File Format**: `.nrrd`
- **Annotations**: **Voxel-level segmentation**
  - `0`: Background
  - `1`: Tumor  

---

## 📥 Installation

Install the package via pip:

```bash
pip install tdsc-abus2023-pytorch
```

### Verify Installation

```python
import tdsc_abus2023_pytorch
print("TDSC-ABUS2023 PyTorch Dataset is installed successfully!")
```

---

## 🚀 Usage

### Loading the Original Dataset

```python
from tdsc_abus2023_pytorch import TDSC, DataSplits

# Initialize dataset with automatic download
dataset = TDSC(
    path="./data",
    split=DataSplits.TRAIN,
    download=True
)

# Access a sample
volume, mask, label, bbx = dataset[0]
```

### Using the Tumor-Only Dataset

This dataset contains only tumor data, suitable for classification and segmentation tasks.

```python
from tdsc_abus2023_pytorch import TDSCTumors, DataSplits

# Initialize dataset with automatic download
dataset = TDSCTumors(
    path="./data",
    split=DataSplits.TRAIN,
    download=True
)

# Access a sample
volume, mask, label = dataset[0]
```

### Data Transformers for Preprocessing

```python
from tdsc_abus2023_pytorch import TDSC, DataSplits
from enum import Enum
import numpy as np

class ViewTransformer:
    class View(Enum):
        CORONAL = 0
        SAGITTAL = 1
        AXIAL = 2
    
    TRANSPOSE_CONFIGS = {
        View.AXIAL: (0, 1, 2),
        View.CORONAL: (1, 2, 0),
        View.SAGITTAL: (2, 0, 1)
    }
    
    def __init__(self, view: View):
        self.transpose_axes = self.TRANSPOSE_CONFIGS[view]
    
    def __call__(self, vol: np.ndarray, mask: np.ndarray):
        transformed_vol = np.transpose(vol, self.transpose_axes)
        transformed_mask = np.transpose(mask, self.transpose_axes)
        return transformed_vol, transformed_mask

view_transformer = ViewTransformer(view=ViewTransformer.View.AXIAL)
dataset = TDSC(path="./data", split=DataSplits.TRAIN, transforms=[view_transformer])

# Get transformed sample
vol, msk, label, bbx = dataset[0]
```

---

## 📂 Data Structure

```
data/
  ├── Train/
  │   ├── DATA/
  │   └── MASK/
  ├── Validation/
  │   ├── DATA/
  │   └── MASK/
  └── Test/
      ├── DATA/
      └── MASK/
```

---

## 📖 Citation

If you use this dataset in your research, please cite:

```bibtex
@misc{luo2025tumordetectionsegmentationclassification,
    title={Tumor Detection, Segmentation and Classification Challenge on Automated 3D Breast Ultrasound: The TDSC-ABUS Challenge},
    author={Gongning Luo and others},
    year={2025},
    eprint={2501.15588},
    archivePrefix={arXiv},
    primaryClass={eess.IV},
    url={https://arxiv.org/abs/2501.15588},
}
```

---

## 🤝 Contributing

We welcome contributions! To contribute, please **fork the repository**, make your changes, and submit a **Pull Request**.
