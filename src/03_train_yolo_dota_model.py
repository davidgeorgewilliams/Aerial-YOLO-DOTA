from helpers import read_yaml_config
from ultralytics import YOLO
import math
import torch

# Detect GPUs
num_gpus = torch.cuda.device_count()
device = ",".join([str(i) for i in range(num_gpus)])

# Load configuration
config = read_yaml_config("dota.yaml")
base_batch_size = config["base_batch_size"]
base_lr = config["base_lr"]
epochs = config["epochs"]
save_period = config["save_period"]

# Load a pre-trained YOLOv9e model
# YOLOv9e is the 'extensive' variant, representing the largest and most powerful model in the YOLOv9 family
# Key advantages of YOLOv9e compared to YOLOv8x:
# - 15% reduction in parameters
# - 25% less computational requirements
# - 1.7% improvement in Average Precision (AP)
# This model offers a balance of high accuracy and improved efficiency, making it suitable for
# complex detection tasks where computational resources are less constrained
model = YOLO("yolov9e.pt")

# Scale the learning rate using the square root of the number of GPUs
# This is a more conservative scaling approach than linear scaling, which helps to:
# 1. Maintain training stability with a large number of GPUs
# 2. Mitigate the risk of divergence due to overly large learning rates
# 3. Provide a balance between faster convergence and avoiding overshooting the optimum
scaled_lr = base_lr * math.sqrt(num_gpus)

# Scale the batch size linearly with the number of GPUs
# This allows each GPU to process the same number of images as in single-GPU training,
# effectively increasing the total batch size and potentially speeding up training
scaled_batch_size = base_batch_size * num_gpus

results = model.train(
    data="dota.yaml",
    epochs=epochs,
    save_period=save_period,
    imgsz=640,
    workers=8 * num_gpus,      # Scale number of workers
    batch=scaled_batch_size,   # Scaled batch size
    lr0=scaled_lr,             # Scaled learning rate
    device=device,             # Use all available GPUs
    val=True,                  # Run validation
    plots=True,                # Generate plots
    verbose=True               # Use verbose training
)
