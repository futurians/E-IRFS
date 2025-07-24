# E-IRFS: Exponentially Weighted Instance-Aware Repeat Factor Sampling
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/downloads/release/python-380/)

📌 This repository introduces **E-IRFS**, a general-purpose sampling method that boosts rare-class performance in long-tailed object detection datasets. It integrates seamlessly with YOLOv11 models from the [Ultralytics](https://github.com/ultralytics/ultralytics) framework.

🧪 Designed for any YOLO dataset — plug and train!

---

## 🔍 What is E-IRFS?

E-IRFS enhances standard Instance-Aware Repeat Factor Sampling (IRFS) by using **exponential scaling** to:
- Give rare classes more exposure during training
- Improve mean average precision (mAP) for underrepresented objects
- Benefit lightweight models like YOLOv11-Nano in UAV and real-time scenarios

[**[Paper]**](https://arxiv.org/abs/2503.21893) - Accepted in the 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025).

---

## 📦 Installation

```bash
git clone https://github.com/futurians/E-IRFS.git
cd E-IRFS
pip install -r requirements.txt
```

---

## 🚀 Training with E-IRFS

Use the unified training script:

```bash
python eirfs_train.py \
  --model yolo11n.pt \
  --data ./examples/example_dataset.yaml \
  --alpha 2.0 \
  --threshold 0.0001 \
  --epochs 100 \
  --imgsz 640
```

### Arguments

| Argument     | Description                                      |
|--------------|--------------------------------------------------|
| `--model`    | YOLOv11 model checkpoint (e.g. `yolo11n.pt`)     |
| `--data`     | Path to dataset YAML config                      |
| `--alpha`    | Exponential scaling factor (default: 2.0)        |
| `--threshold`| Sampling threshold for rebalancing (default: 0.0001) |
| `--epochs`   | Number of training epochs                        |
| `--imgsz`    | image size (default: 640)                         |

---

## 📁 Example Dataset YAML

```yaml
train: path/to/train/images
val: path/to/val/images
nc: 4
names: ['ClassA', 'ClassB', 'ClassC', 'ClassD']
```

You can use E-IRFS with any YOLOv11-compatible dataset.


---

## 📊 Benchmark Datasets in EIRFS

To ensure transparency and reproducibility, we provide detailed metadata for every image used in our training and validation phases. You can now explore:

📁 [`train_dataset.csv`](https://github.com/futurians/E-IRFS/blob/main/train_dataset.csv)
📁 [`validation_dataset.csv`](https://github.com/futurians/E-IRFS/blob/main/validation_dataset.csv)

Each CSV contains:

* Image filename and corresponding annotation file name

These files enable easy tracking, auditing, and reproduction of our training pipeline.

---

### 📦 Datasets Used

The following open-source datasets were used in this benchmark. You may download them individually and reconstruct the combined training/validation sets using the provided CSVs.

| Dataset Name              | Description                                                       | Download Link                                                                                             | Citation Key                          |
| ------------------------- | ----------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| **FireMan-UAV-RGBT**      | RGB-Thermal drone video dataset for wildfire detection in Finland | [Zenodo](https://zenodo.org/records/13732947)                    | `Kularatne2024FireManPaper`           |
| **Forest Fire & Smoke**   | Annotated fire/smoke drone images                                 | [Roboflow](https://universe.roboflow.com/master-candidate/forest-fire-and-smoke-rf4pd)                    | `forest-fire-and-smoke-rf4pd_dataset` |
| **SAUS**                  | Small aerial image dataset with class-labeled objects             | [Roboflow](https://universe.roboflow.com/saus/saus)                                                       | `saus_dataset`                        |
| **SAR Custom Drone**      | Aerial dataset tailored for UAV-based search and rescue           | [Roboflow](https://universe.roboflow.com/university-of-engineering-and-technology-huotg/sar_custom_drone) | `sar_custom_drone_dataset`            |
| **Merged Thesis Dataset** | Aggregated UAV object detection dataset used in prior work        | [Roboflow](https://universe.roboflow.com/diplomatic/merged-thesis)                                        | `merged-thesis_dataset`               |

---

### 🧰 How to Use

1. Download each dataset using the links above.
2. Use the filenames and paths listed in `train_dataset.csv` and `validation_dataset.csv` to:

   * Copy relevant images and labels to your working directory.
   * Maintain the structure expected by YOLO:

     ```
     dataset/
     ├── train/
     │   ├── images/
     │   └── labels/
     └── val/
         ├── images/
         └── labels/
     ```
3. Update your YOLO `.yaml` config to reflect these locations for training with E-IRFS.

---

### 📚 Citations

```bibtex
@INPROCEEDINGS{Kularatne2024FireManPaper,
  author={Kularatne, S.D.M.W. and {\'A}lvarez Casado, Constantino  and Rajala, Janne and Hänninen, Tuomo and Bordallo López, Miguel and Nguyen, Le},
  booktitle={2024 IEEE 29th International Conference on Emerging Technologies and Factory Automation (ETFA)}, 
  title={FireMan-UAV-RGBT: A Novel UAV-Based RGB-Thermal Video Dataset for the Detection of Wildfires in the Finnish Forests}, 
  year={2024},
  pages={1-8}
}

@misc{forest-fire-and-smoke-rf4pd_dataset,
  title = { forest fire and smoke Dataset },
  type = { Open Source Dataset },
  howpublished = {\url{https://universe.roboflow.com/master-candidate/forest-fire-and-smoke-rf4pd}},
  journal = {Roboflow Universe },
  publisher = {Roboflow },
  year = {2024 },
  month = {jan },
  note = { visited on 2025-02-15 },
}

@misc{saus_dataset,
  title = {Saus Dataset},
  author = {Saus},
  year = {2024},
  month = {jun},
  howpublished = {\url{https://universe.roboflow.com/saus/saus}},
  journal = {Roboflow Universe},
  publisher = {Roboflow},
  note = {Visited on 2025-02-15},
}

@misc{sar_custom_drone_dataset,
  title = {SAR Custom DRONE Dataset},
  author = {University of Engineering and Technology},
  year = {2023},
  month = {dec},
  howpublished = {\url{https://universe.roboflow.com/university-of-engineering-and-technology-huotg/sar_custom_drone}},
  journal = {Roboflow Universe},
  publisher = {Roboflow},
  note = {Visited on 2025-02-15},
}

@misc{merged-thesis_dataset,
  title = {Merged Thesis Dataset},
  author = {Diplomatic},
  year = {2023},
  month = {jan},
  howpublished = {\url{https://universe.roboflow.com/diplomatic/merged-thesis}},
  journal = {Roboflow Universe},
  publisher = {Roboflow},
  note = {Visited on 2025-02-15},
}
```

---

## 📈 Citation

If you use E-IRFS in your work, please cite our IROS 2025 paper:

```bibtex
@misc{ahmed2025exponentiallyweightedinstanceawarerepeat,
      title={Exponentially Weighted Instance-Aware Repeat Factor Sampling for Long-Tailed Object Detection Model Training in Unmanned Aerial Vehicles Surveillance Scenarios}, 
      author={Taufiq Ahmed and Abhishek Kumar and Constantino Álvarez Casado and Anlan Zhang and Tuomo Hänninen and Lauri Loven and Miguel Bordallo López and Sasu Tarkoma},
      year={2025},
      eprint={2503.21893},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.21893}, 
}
```

---

## 📬 Contact
For questions or collaborations, please reach out:    
🌐 https://github.com/futurians

---

## 🧪 Acknowledgements

This research is supported by:
- Research Council of Finland 6G Flagship (Grant No. 369116)
- FIREMAN Project (Grant No. 348008)
- Business Finland Neural Publish-Subscribe for 6G Project

