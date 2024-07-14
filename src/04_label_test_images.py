import argparse
import cv2
from helpers import read_yaml_config
from ultralytics import YOLO


def draw_boxes(image, boxes, class_names):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf)
        cls = int(box.cls)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_names[cls]} {conf:.2f}"

        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Draw filled rectangle for text background
        cv2.rectangle(image, (x1, y1 - text_height - 5), (x1 + text_width, y1), (0, 255, 0), -1)

        # Put text on the filled rectangle
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return image


def predict_and_draw(model_path, image_path, yaml_path, output_path):
    # Load the model
    model = YOLO(model_path)

    # Load class names from YAML
    yaml_config = read_yaml_config(yaml_path)
    class_names = yaml_config['names']

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image at {image_path}")

    # Run inference
    results = model(image)

    # Draw bounding boxes
    annotated_image = draw_boxes(image, results[0].boxes, class_names)

    # Save the annotated image
    cv2.imwrite(output_path, annotated_image)
    print(f"Annotated image saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO inference on an image and draw bounding boxes")
    parser.add_argument("model_path", help="Path to the YOLO model weights")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("yaml_path", help="Path to the YAML configuration file")
    parser.add_argument("output_path", help="Path to save the annotated image")

    args = parser.parse_args()

    predict_and_draw(args.model_path, args.image_path, args.yaml_path, args.output_path)
