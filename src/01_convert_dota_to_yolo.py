import os
import cv2
import argparse
from tqdm import tqdm

# Set up argument parser
parser = argparse.ArgumentParser(description='Convert DOTA labels to YOLO format.')
parser.add_argument('--data_root', type=str, required=True,
                    help='Root directory of the DOTA dataset')

# Parse arguments
args = parser.parse_args()

# Use the parsed data_root
data_root = args.data_root

# Validate the data root
if not os.path.isdir(data_root):
    raise ValueError(f"The specified data root directory does not exist: {data_root}")

# Define paths
image_dir = os.path.join(data_root, "images")
label_dir = os.path.join(data_root, "labels")
yolo_label_dir = os.path.join(data_root, "yolo_labels")

# Ensure the YOLO label directory exists
os.makedirs(yolo_label_dir, exist_ok=True)


def convert_dota_to_yolo(dota_label_path, image_path, output_path):
    # Read image dimensions
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return
    img_height, img_width = img.shape[:2]

    # Define class mapping (adjust this based on your DOTA classes)
    class_mapping = {
        "baseball-diamond": 0,
        "basketball-court": 1,
        "bridge": 2,
        "ground-track-field": 3,
        "harbor": 4,
        "helicopter": 5,
        "large-vehicle": 6,
        "plane": 7,
        "roundabout": 8,
        "ship": 9,
        "small-vehicle": 10,
        "soccer-ball-field": 11,
        "storage-tank": 12,
        "swimming-pool": 13,
        "tennis-court": 14
    }

    yolo_lines = []

    with open(dota_label_path, "r") as f:
        lines = f.readlines()

    for line in lines[2:]:  # Skip the first two lines (imagesource and gsd)
        parts = line.strip().split()
        if len(parts) < 10:  # Ensure we have enough parts
            raise ValueError(f"Not enough parts: {line}")

        # Extract coordinates and class
        coords = [float(p) for p in parts[:8]]
        class_name = parts[8].lower()

        # Calculate center, width, and height
        x_coords, y_coords = coords[::2], coords[1::2]
        center_x = sum(x_coords) / 4
        center_y = sum(y_coords) / 4
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)

        # Normalize coordinates
        center_x /= img_width
        center_y /= img_height
        width /= img_width
        height /= img_height

        # Get class ID
        class_id = class_mapping.get(class_name, -1)
        if class_id == -1:
            print(f"Warning: Unknown class {class_name}")
            continue

        # Create YOLO format line
        yolo_line = f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n"
        yolo_lines.append(yolo_line)

    # Write YOLO format labels
    with open(output_path, "w") as f:
        f.writelines(yolo_lines)


# Get the total number of label files
total_files = sum(1 for file in os.listdir(label_dir) if file.endswith(".txt"))

# Convert all labels with progress bar
with tqdm(total=total_files, desc="Converting labels", unit="file") as pbar:
    for label_file in os.listdir(label_dir):
        if label_file.endswith(".txt"):
            image_file = label_file.replace(".txt", ".png")

            # Update progress bar description with current file
            pbar.set_description(f"Converting {label_file}")

            # Convert DOTA labels to YOLO format
            convert_dota_to_yolo(
                os.path.join(label_dir, label_file),
                os.path.join(image_dir, image_file),
                os.path.join(yolo_label_dir, label_file))

            # Update progress bar
            pbar.update(1)

print("Conversion completed.")
