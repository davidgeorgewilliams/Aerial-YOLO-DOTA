import argparse
import os

import cv2
from tqdm import tqdm
from ultralytics import YOLO

from helpers import read_yaml_config


def draw_boxes(image, boxes, class_names):
    """
    Draw bounding boxes and labels on the image.

    Args:
    image (numpy.ndarray): The input image.
    boxes (list): List of detected boxes from YOLO.
    class_names (list): List of class names corresponding to class indices.

    Returns:
    numpy.ndarray: The image with bounding boxes and labels drawn.
    """
    for box in boxes:
        # Extract bounding box coordinates and convert to integers
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Get the confidence score and class index
        conf = float(box.conf)
        cls = int(box.cls)

        # Draw the bounding box on the image
        # Color is set to green (0, 255, 0) with a thickness of 2
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Create the label text with class name and confidence score
        label = f"{class_names[cls]} {conf:.2f}"

        # Get the size of the label text
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Draw a filled rectangle as background for the label
        # This improves readability of the label
        cv2.rectangle(image, (x1, y1 - text_height - 5), (x1 + text_width, y1), (0, 255, 0), -1)

        # Put the label text on the image
        # Text is black (0, 0, 0) for contrast against the green background
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return image


def predict_and_draw(model, image_path, class_names, output_path):
    """
    Load an image, run YOLO prediction, draw bounding boxes, and save the result.

    Args:
    model (YOLO): The loaded YOLO model.
    image_path (str): Path to the input image file.
    class_names (list): List of class names corresponding to class indices.
    output_path (str): Path where the annotated image will be saved.

    Returns:
    None
    """
    # Load the image from the specified path
    image = cv2.imread(image_path)

    # Check if the image was successfully loaded
    if image is None:
        print(f"Unable to read image at {image_path}")
        return

    # Run YOLO prediction on the image
    # The model returns a list of Results objects, one for each image
    results = model(image)

    # Draw bounding boxes on the image
    # We use results[0] as we're only processing one image
    # results[0].boxes contains all the detected boxes for this image
    annotated_image = draw_boxes(image, results[0].boxes, class_names)

    # Save the annotated image to the specified output path
    cv2.imwrite(output_path, annotated_image)


def process_test_images(model_path):
    """
    Process all test images in the dataset using a YOLO model.

    Args:
    model_path (str): Path to the YOLO model weights file.

    Returns:
    None
    """
    # Load configuration from the YAML file
    config = read_yaml_config("dota.yaml")
    data_root = config["path"]  # Root directory of the dataset
    class_names = config['names']  # List of class names

    # Define paths for different directories
    image_dir = os.path.join(data_root, "images")  # Directory containing all images
    label_dir = os.path.join(data_root, "labels")  # Directory containing label files
    annotated_dir = os.path.join(data_root, "annotated_test_images")  # Directory to save annotated images
    os.makedirs(annotated_dir, exist_ok=True)  # Create annotated directory if it doesn't exist

    # Load the YOLO model
    model = YOLO(model_path)

    # Iterate over all images in the image directory
    for img_name in tqdm(os.listdir(image_dir), desc="Processing images"):
        # Check if the file is a PNG image
        if img_name.lower().endswith('.png'):
            img_path = os.path.join(image_dir, img_name)
            # Construct the path for the corresponding label file
            label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + '.txt')

            # Check if it's a test image (no corresponding label file)
            if not os.path.exists(label_path):
                # Construct the output path for the annotated image
                output_path = os.path.join(annotated_dir, img_name)
                # Process the image: predict bounding boxes and draw them
                predict_and_draw(model, img_path, class_names, output_path)

    print(f"Annotated images saved to {annotated_dir}")


if __name__ == "__main__":
    # This block only executes if the script is run directly (not imported as a module)

    # Set up the argument parser for command-line arguments
    parser = argparse.ArgumentParser(description="Run YOLO inference on test images and draw bounding boxes")

    # Add argument for the path to the YOLO model weights file
    parser.add_argument("--model_path", required=True, help="Path to the YOLO model weights")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function to process test images
    # Pass the path to the model weights
    process_test_images(args.model_path)
