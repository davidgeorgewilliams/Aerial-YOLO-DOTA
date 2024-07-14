# Aerial-YOLO-DOTA: Advanced Object Detection for Aerial Imagery

This repository showcases state-of-the-art object detection in aerial and satellite imagery using [YOLOv9](https://docs.ultralytics.com/models/yolov9/) on the [DOTA (Dataset for Object Detection in Aerial Images) v1.5](https://captain-whu.github.io/DOTA/dataset.html) dataset.

![Aerial-YOLO-DOTA.jpg](docs/Aerial-YOLO-DOTA.jpg)

## Key Features

- Implementation of [YOLOv9](https://docs.ultralytics.com/models/yolov9/) for efficient, accurate detection of small objects in large-scale aerial images
- Custom data pipeline for the [DOTA v1.5](https://captain-whu.github.io/DOTA/dataset.html) dataset, supporting rotated bounding boxes
- Fine-tuned models achieving high performance on aerial object categories
- Interactive demo for real-time object detection on aerial imagery
- Comprehensive evaluation metrics and visualization tools

## Experiments

We conducted three main experiments with varying hyperparameters and computational resources:

1. **100e-16b-0.01lr**
   - 100 epochs, 16 batch size, 0.01 learning rate
   - Trained on 1x H100 (80 GB PCIe) 26 CPU cores, 205.4 GB RAM, 1.1 TB SSD

2. **250e-128b-0.028lr**
   - 250 epochs, 128 batch size, 0.028 learning rate
   - Trained on 8x H100 (80 GB SXM5) 208 CPU cores, 1.9 TB RAM, 24.2 TB SSD

3. **350e-256b-0.0028**
   - 350 epochs, 256 batch size, 0.0028 learning rate
   - Trained on 8x H100 (80 GB SXM5) 208 CPU cores, 1.9 TB RAM, 24.2 TB SSD

## Results

For each experiment, we provide the following performance metrics and visualizations:

- Normalized Confusion Matrix
- F1 Score Curve
- Precision Curve
- Precision-Recall Curve
- Recall Curve
- Overall Results Summary

These can be found in the `results` directory, organized by experiment name.

### Best Model Performance

(Insert a brief discussion of which model performed best and why)

### Test Set Annotations

We've included a gallery of annotations from the test dataset using our best-performing model. These images demonstrate the model's ability to detect and classify objects in real-world aerial imagery scenarios.

## Dataset

We have prepared an easy-to-use consolidated DOTA 1.5 dataset, including labels suitable for use with [YOLOv9](https://docs.ultralytics.com/models/yolov9/). This dataset is available on [Google Drive](insert_link_here).

## Repository Structure

```plaintext
Aerial-YOLO-DOTA/
├── src/
│   ├── 01_convert_dota_to_yolo.py
│   ├── 02_create_train_test_split.py
│   ├── 03_train_yolo_dota_model.py
│   ├── 04_label_test_images.py
│   └── helpers.py
├── results/
│   ├── 100e-16b-0.01lr/
│   ├── 250e-128b-0.028lr/
│   └── 350e-256b-0.0028lr/
├── annotated_test_images/
├── dota.yaml
└── README.md
```

## Usage

Clone this repository:
```shell
git clone https://github.com/your_username/Aerial-YOLO-DOTA.git
```

Install dependencies:
```shell
pip install -r requirements.txt
```

Download the [DOTA v1.5](https://captain-whu.github.io/DOTA/dataset.html) dataset from [our Google Drive link](insert_link_here) and place it in the appropriate directory.

Change directory into `src`:
```shell
cd src
```
Our scripts assume you are running from within the src directory. All subsequent commands should be executed from this location.

Convert DOTA labels to YOLO format:

```shell
python 01_convert_dota_to_yolo.py
```
However, this step is not necessary if you're using our pre-processed dataset.

Create train/test split:
```shell
python 02_create_train_test_split.py
```

Train the model:
```shell
python 03_train_yolo_dota_model.py 
```

Label test images:
```shell
python 04_label_test_images.py --model path/to/best.pt
```

## Conclusion

This project demonstrates the effectiveness of [YOLOv9](https://docs.ultralytics.com/models/yolov9/) in detecting objects in aerial imagery using the [DOTA v1.5](https://captain-whu.github.io/DOTA/dataset.html) dataset. Our experiments show (insert brief conclusion about model performance and insights gained).

The codebase provided here offers a complete pipeline for training, evaluating, and using [YOLOv9](https://docs.ultralytics.com/models/yolov9/) models on aerial imagery datasets, and can serve as a strong foundation for further research or practical applications in this domain.

## Acknowledgements

We thank the creators of the [DOTA](https://captain-whu.github.io/DOTA/) dataset and the developers of [YOLOv9](https://docs.ultralytics.com/models/yolov9/) for their invaluable contributions to the field of object detection in aerial imagery.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
