# TDSC-ABUS2023 PyTorch Dataset

[![PyPI version](https://img.shields.io/pypi/v/tdsc-abus2023-pytorch)](https://pypi.org/project/tdsc-abus2023-pytorch/)  
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)  
[![Build Status](https://img.shields.io/github/actions/workflow/status/your-repo/ci.yml?branch=main)](https://github.com/your-repo/actions)

A PyTorch-compatible dataset package containing volumetric data from the **TDSC-ABUS2023** collection (**Tumor Detection, Segmentation, and Classification Challenge on Automated 3D Breast Ultrasound**).

## Dataset Description

The dataset consists of **200 3D ultrasound volumes** collected using an **Invenia ABUS (GE Healthcare)** system at **Harbin Medical University Cancer Hospital, China**.  
All tumor annotations were created and verified by experienced radiologists.

### 📊 Dataset Composition

| **Set**       | **Cases** | **Malignant** | **Benign** |
|--------------|----------|--------------|------------|
| **Training**  | 100      | 58           | 42         |
| **Validation**| 30       | 17           | 13         |
| **Test**      | 70       | 40           | 30         |

### 📌 Technical Specifications
- **Image Dimensions**: Varying between **843×546×270** and **865×682×354**  
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

```python
from tdsc_abus2023_pytorch import TDSC, TDSCTumors, DataSplits

# Initialize dataset with automatic download
dataset = TDSC(
    path="./data",
    split=DataSplits.TRAIN,
    download=True
)

# Access a sample
volume, mask, label, bbx = dataset[0]
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

We welcome contributions!  
To contribute, please **fork the repository**, make your changes, and submit a **Pull Request**.