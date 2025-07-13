import numpy as np
from ultralytics.data.dataset import YOLODataset
import ultralytics.data.build as build

import numpy as np
from ultralytics.data.dataset import YOLODataset
import ultralytics.data.build as build

EIRFS_THRESHOLD = 0.0001  # Global threshold

class YOLOEIRFSDataset(YOLODataset):
    def __init__(self, *args, mode="train", alpha=2.0, t=EIRFS_THRESHOLD, **kwargs):
        super().__init__(*args, **kwargs)

        self.train_mode = "train" in self.prefix
        self.alpha = alpha
        self.t = t

        self.num_classes = len(self.data["names"])
        self.image_counts = np.ones(self.num_classes, dtype=np.float32)
        self.instance_counts = np.ones(self.num_classes, dtype=np.float32)

        self._count_class_frequencies()
        self._compute_sampling_weights()

    def _count_class_frequencies(self):
        """Count how often each class appears across images and instances."""
        seen_in_image = [set() for _ in range(self.num_classes)]
        for label in self.labels:
            cls_ids = label['cls'].astype(int).flatten()
            unique = np.unique(cls_ids)
            for c in unique:
                seen_in_image[c].add(id(label))  # ensure image-level uniqueness
            for c in cls_ids:
                self.instance_counts[c] += 1

        self.image_counts = np.array([len(s) if len(s) > 0 else 1 for s in seen_in_image], dtype=np.float32)
        self.instance_counts = np.where(self.instance_counts == 0, 1, self.instance_counts)

    def _compute_sampling_weights(self):
        """Compute class weights and per-image sampling probabilities."""
        f_ic = self.image_counts / np.sum(self.image_counts)
        f_bc = self.instance_counts / np.sum(self.instance_counts)
        self.class_weights = np.exp(self.alpha * np.sqrt(self.t / np.sqrt(f_ic * f_bc)))

        self.weights = np.array([
            np.max(self.class_weights[label['cls'].astype(int).flatten()])
            if label['cls'].size else 1.0
            for label in self.labels
        ])

        total = np.sum(self.weights)
        self.probabilities = self.weights / total if total > 0 else np.ones_like(self.weights) / len(self.weights)

    def __getitem__(self, index):
        """Sample images based on precomputed probabilities (E-IRFS) during training."""
        if self.train_mode:
            index = np.random.choice(len(self.labels), p=self.probabilities)
        return self.transforms(self.get_image_and_label(index))

# Override Ultralytics dataset with this E-IRFS variant
build.YOLODataset = YOLOEIRFSDataset


# Example usage with configurable input
if __name__ == "__main__":
    import argparse
    from ultralytics import YOLO

    parser = argparse.ArgumentParser(description="Run E-IRFS with different alpha values")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLOv11 model")
    parser.add_argument("--data", type=str, required=True, help="Dataset YAML path")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--alphas", type=float, nargs="+", default=[2.0],
                        help="List of alpha values to run E-IRFS ablation")
    parser.add_argument("--threshold", type=float, default=0.0001, help="E-IRFS repeat factor threshold")

    args = parser.parse_args()

    for alpha in args.alphas:
        print(f"Running experiment with alpha = {alpha}")
        build.YOLODataset = lambda *a, **kw: YOLOEIRFSDataset(*a, alpha=alpha, t=args.threshold, **kw)

        model = YOLO(args.model)
        model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz)
