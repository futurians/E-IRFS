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

