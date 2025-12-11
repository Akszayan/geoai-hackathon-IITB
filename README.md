# **GeoAI Hackathon — Theme 1: Drone Imagery Feature Extraction**

### **DeepLabV3-ResNet50 Semantic Segmentation Pipeline (Punjab District)**

---

## **🏅 Team Information**

**Team Name:** AeroMappers        **Team ID:** Nati-250309
---

## **📌 Badges**

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5-orange)
![DeepLabV3](https://img.shields.io/badge/Model-DeepLabV3--ResNet50-green)
![CUDA](https://img.shields.io/badge/CUDA-12.1-success)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)
![Dataset: PB District](https://img.shields.io/badge/Dataset-Punjab%20(PB)-brightgreen)
![Hackathon](https://img.shields.io/badge/National%20GeoAI%20Hackathon-2025-red)


---

# **1. Overview**

This repository contains the full codebase developed for **Round‑1** of the **National Geo-AI Hackathon 2025 (Theme 1)** — *Automated Feature Extraction from Drone Imagery*.

We built a **6‑class semantic segmentation model** capable of extracting:

* Building footprints
* Roof‑material classes (Roof_type 1–4)
* Water bodies
* Background land (class 0)

The output is a fully GIS-compatible set of **raster masks** and **polygon vectors**, validated using QGIS.

---

# **2. Pipeline Summary**

### **Core Stages**

1. **Shapefile cleaning + reprojection (EPSG:32643)**
2. **Raster mask generation (6 classes)**
3. **512×512 tile creation** for ortho + masks
4. **DeepLabV3-ResNet50 training** (ImageNet backbone, AMP enabled)
5. **Full‑ortho inference with tiling**
6. **Polygonization + area-based cleanup**
7. **QGIS overlay verification**

---

# **3. Repository Structure**

```
geoai-hackathon-IITB/
│
├── raw_data/                     # Orthos + shapefiles (PB)
│   ├── orthos/
│   └── shp/
│
├── data/
│   └── meta/                     # train_tiles.txt, val_tiles.txt
│
├── scripts/
│   ├── generate_tiles_npy.py     # Create .npy tiles
│   ├── train_deeplabv3_resumable.py
│   ├── infer_full_ortho.py       # Predict full orthomosaics
│   ├── polygonize_mask.py
│   ├── postprocess_polygons.py
│   └── visualize_overlay.py
│
├── models/
│   ├── deeplabv3_pb_best.pth     # Best model
│   └── deeplabv3_pb_last.pth
│
├── outputs/
│   ├── predictions/              # Full-ortho prediction masks
│   ├── vectors/                  # Raw polygons
│   ├── vectors_clean/            # Cleaned polygons
│   └── figs/                     # Visual overlays
│
├── configs/
│   └── train_deeplabv3_pb.yaml   # Training configuration
│
├── README.md
├── .gitignore
├── requirements.txt
└── LICENSE
```

---

# **4. Model Details**

### **Model:** DeepLabV3 + ResNet50 backbone

* ASPP module for multi‑scale context
* Pretrained on ImageNet
* Fine‑tuned for **6 output classes**

### **Training Setup**

| Component | Value             |
| --------- | ----------------- |
| Loss      | Cross‑entropy     |
| Optimizer | AdamW (lr: 1e‑4)  |
| Scheduler | CosineAnnealingLR |
| AMP       | Enabled           |
| Epochs    | 30                |
| GPU       | RTX 4060 Laptop   |

### **Validation Performance**

| Class    | IoU        |
| -------- | ---------- |
| Roof 1   | 0.845      |
| Roof 2   | 0.763      |
| Roof 3   | 0.322      |
| Roof 4   | 0.610      |
| Water    | 0.546      |
| **mIoU** | **0.5583** |

---

# **5. Usage**

### **Training**

```
python scripts/train_deeplabv3_resumable.py --config configs/train_deeplabv3_pb.yaml
```

### **Inference on full orthomosaic**

```
python scripts/infer_full_ortho.py \
  --ortho <path_to_ortho> \
  --checkpoint models/deeplabv3_pb_best.pth \
  --out outputs/predictions/ \
  --device cuda
```

### **Polygonization**

```
python scripts/polygonize_mask.py --mask <pred_mask.tif> --out outputs/vectors/
```

---

# **6. Output Examples**

---

# **7. License**

Released under the **MIT License**.

---

# **8. Acknowledgements**

This project is developed for the **National Geo-AI Hackathon 2025** organized under Techfest IITB.

---


