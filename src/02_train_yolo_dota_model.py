from ultralytics import YOLO
import torch

# Load a pre-trained YOLOv9 model
model = YOLO("yolov9e.pt")

num_gpus = torch.cuda.device_count()
base_batch_size = 16
base_lr = 0.01

results = model.train(
    data="dota.yaml",
    epochs=100,
    imgsz=640,
    batch=base_batch_size * num_gpus,   # Scale batch size
    workers=8 * num_gpus,               # Scale number of workers
    lr0=base_lr * num_gpus,             # Scale learning rate
    device="",                          # This will use all available GPUs
    split=0.1,                          # 10% of data used for validation
    save_period=10,
    metrics=[
        "mAP50-95",     # Primary metric, mean Average Precision over IoU thresholds
        "mAP50",        # Mean Average Precision at IoU=0.50
        "precision",    # Precision
        "recall",       # Recall
        "f1",           # F1-score (harmonic mean of precision and recall)
        "fitness",      # A combined metric of mAP and speed
        "box_loss",     # Loss for bounding box regression
        "cls_loss",     # Loss for class prediction
        "speed",        # Inference speed (images per second)
    ]
)
