import os
import shutil

from sklearn.model_selection import train_test_split

from helpers import read_yaml_config


def create_train_val_splits(yaml_path, train_ratio=0.8):
    """
    Create train and validation splits for a dataset based on YAML configuration.

    This function reads a YAML configuration file, splits the dataset into training
    and validation sets, and organizes the files into appropriate directories.

    Args:
        yaml_path (str): Path to the YAML configuration file.
        train_ratio (float, optional): Ratio of data to use for training. Defaults to 0.8.

    Returns:
        None

    Raises:
        FileNotFoundError: If the YAML file, image files, or label files are not found.
        yaml.YAMLError: If there's an error parsing the YAML file.
        OSError: If there's an error creating directories or moving files.

    Notes:
        - The function assumes that images and labels have matching filenames
          (excluding extensions).
        - Image files are assumed to be .png format. Adjust if using a different format.
        - The function creates 'train' and 'val' subdirectories in both 'images' and
          'yolo_labels' directories.
        - Files are moved, not copied. Ensure you have backups if needed.

    Example:
        >>> create_train_val_splits('config.yaml', train_ratio=0.75)
        Moved 750 files to train and 250 files to val.

    Warning:
        This function modifies the directory structure. Ensure you have necessary
        permissions and backups before running.
    """
    # Read configuration from YAML file
    config = read_yaml_config(yaml_path)
    data_root = config["path"]

    image_dir = os.path.join(data_root, "images")
    label_dir = os.path.join(data_root, "labels")
    # Get all label files
    all_labels = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

    # Split into train and val
    train_labels, val_labels = train_test_split(all_labels, train_size=train_ratio, random_state=42)

    # Create train and val directories
    os.makedirs(os.path.join(data_root, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(data_root, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(data_root, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(data_root, "labels", "val"), exist_ok=True)

    # Move files to train and val directories
    for label in train_labels:
        # Move label file
        shutil.move(str(os.path.join(label_dir, label)), str(os.path.join(data_root, "labels", "train", label)))

        # Move corresponding image file
        img_name = label.rsplit(".", 1)[0] + ".png"  # Assuming .png, adjust if needed
        if os.path.exists(os.path.join(image_dir, img_name)):
            shutil.move(str(os.path.join(image_dir, img_name)),
                        str(os.path.join(data_root, "images", "train", img_name)))
        else:
            print(f"Warning: Image file {img_name} not found for label {label}")

    for label in val_labels:
        # Move label file
        shutil.move(str(os.path.join(label_dir, label)), str(os.path.join(data_root, "labels", "val", label)))

        # Move corresponding image file
        img_name = label.rsplit(".", 1)[0] + ".png"  # Assuming .png, adjust if needed
        if os.path.exists(os.path.join(image_dir, img_name)):
            shutil.move(str(os.path.join(image_dir, img_name)), str(os.path.join(data_root, "images", "val", img_name)))
        else:
            print(f"Warning: Image file {img_name} not found for label {label}")

    print(f"Moved {len(train_labels)} files to train and {len(val_labels)} files to val.")


# Execute the train-validation split function
# This function reads the configuration from 'dota.yaml',
# splits the dataset into 80% training and 20% validation sets,
# and updates the YAML file with the new data organization.
# The split is based on available labeled data to ensure all
# training and validation samples have corresponding labels.
create_train_val_splits("dota.yaml", train_ratio=0.8)
