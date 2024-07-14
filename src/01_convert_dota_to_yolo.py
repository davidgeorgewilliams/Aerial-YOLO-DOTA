from tqdm import tqdm
import os
import cv2
import argparse

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
dota_label_dir = os.path.join(data_root, "dota_labels")
yolo_label_dir = os.path.join(data_root, "labels")

# Ensure the YOLO label directory exists
os.makedirs(yolo_label_dir, exist_ok=True)


def convert_dota_to_yolo(dota_label_path, image_path, output_path):
    """
    Convert DOTA dataset annotations to YOLO format.

    This function reads DOTA format annotations, processes them, and converts
    them to YOLO format. It handles the conversion of rotated bounding boxes
    to axis-aligned bounding boxes and normalizes the coordinates.

    Args:
        dota_label_path (str): Path to the input DOTA format label file.
        image_path (str): Path to the corresponding image file.
        output_path (str): Path where the converted YOLO format label will be saved.

    Returns:
        None

    Raises:
        ValueError: If a line in the DOTA label file has insufficient parts or
                    if an unknown object class is encountered.

    Notes:
        - The function skips the first two lines of the DOTA label file (imagesource and gsd).
        - It calculates the center, width, and height of the bounding box and normalizes them.
        - Coordinates that fall outside the image boundaries are warned about and skipped.
        - The YOLO format output is: <class_id> <center_x> <center_y> <width> <height>,
          where all values are normalized to [0, 1].

    Warning:
        This function assumes that the DOTA labels use absolute pixel coordinates.
        Ensure that the image dimensions match the coordinate system used in annotations.
    """
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
        "container-crane": 3,
        "ground-track-field": 4,
        "harbor": 5,
        "helicopter": 6,
        "large-vehicle": 7,
        "plane": 8,
        "roundabout": 9,
        "ship": 10,
        "small-vehicle": 11,
        "soccer-ball-field": 12,
        "storage-tank": 13,
        "swimming-pool": 14,
        "tennis-court": 15
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
            raise ValueError(f"Unknown class {class_name}")

        if 0 <= center_x <= 1 and 0 <= center_y <= 1 and 0 < width <= 1 and 0 < height <= 1:
            yolo_line = f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n"
            yolo_lines.append(yolo_line)
        else:
            print(f"Warning: Invalid normalized coordinates for"
                  f" {image_path}: {center_x}, {center_y}, {width}, {height}")
            continue

        # Create YOLO format line
        yolo_line = f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n"
        yolo_lines.append(yolo_line)

    # Write YOLO format labels
    with open(output_path, "w") as f:
        f.writelines(yolo_lines)


# Get the total number of label files
total_files = sum(1 for file in os.listdir(dota_label_dir) if file.endswith(".txt"))

# Convert all labels with progress bar
with tqdm(total=total_files, desc="Converting labels", unit="file") as pbar:
    for label_file in os.listdir(dota_label_dir):
        if label_file.endswith(".txt"):
            image_file = label_file.replace(".txt", ".png")

            # Update progress bar description with current file
            pbar.set_description(f"Converting {label_file}")

            # Convert DOTA labels to YOLO format
            convert_dota_to_yolo(
                os.path.join(dota_label_dir, label_file),
                os.path.join(image_dir, image_file),
                os.path.join(yolo_label_dir, label_file))

            # Update progress bar
            pbar.update(1)

print("Conversion completed.")
