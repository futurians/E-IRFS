import numpy as np
from ultralytics.data.dataset import YOLODataset
import ultralytics.data.build as build

# Define a global threshold for E-IRFS
EIRFS_THRESHOLD = 0.0001  # Can be overridden by user

class YOLOEIRFSDataset(YOLODataset):
    def __init__(self, *args, mode="train", alpha=2.0, t=EIRFS_THRESHOLD, **kwargs):
        super(YOLOEIRFSDataset, self).__init__(*args, **kwargs)

        self.train_mode = "train" in self.prefix
        self.alpha = alpha
        self.t = t

        self.count_instances()
        self.class_weights = self.calculate_class_weights(alpha=self.alpha)
        self.weights = self.calculate_weights()
        self.probabilities = self.calculate_probabilities()

    def count_instances(self):
        self.image_counts = [0 for _ in range(len(self.data["names"]))]
        self.instance_counts = [0 for _ in range(len(self.data["names"]))]

        for label in self.labels:
            cls = label['cls'].reshape(-1).astype(int)
            unique_classes = set(cls)
            for class_id in unique_classes:
                self.image_counts[class_id] += 1
            for class_id in cls:
                self.instance_counts[class_id] += 1

        self.image_counts = np.where(self.image_counts == 0, 1, self.image_counts)
        self.instance_counts = np.where(self.instance_counts == 0, 1, self.instance_counts)

    def calculate_class_weights(self, alpha=2.0):
        f_ic = self.image_counts / np.sum(self.image_counts)
        f_bc = self.instance_counts / np.sum(self.instance_counts)
        return np.exp(alpha * np.sqrt(self.t / np.sqrt(f_ic * f_bc)))

    def calculate_weights(self):
        weights = []
        for label in self.labels:
            cls = label['cls'].reshape(-1).astype(int)
            if cls.size == 0:
                weights.append(1)
                continue
            weight = np.max(self.class_weights[cls])
            weights.append(weight)
        return weights

    def calculate_probabilities(self):
        total_weight = sum(self.weights)
        return [w / total_weight for w in self.weights]

    def __getitem__(self, index):
        if not self.train_mode:
            return self.transforms(self.get_image_and_label(index))
        else:
            index = np.random.choice(len(self.labels), p=self.probabilities)
            return self.transforms(self.get_image_and_label(index))

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
