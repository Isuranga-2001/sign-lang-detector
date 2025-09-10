# Sign Language Detector

Real-time American Sign Language (ASL) gesture detection using a YOLOv8 model. This project is trained to recognize and classify ASL signs from images or video input, leveraging the YOLOv8 architecture for fast and accurate detection.

## Dataset Setup

1. Download the dataset from [this link](https://1drv.ms/f/c/41b2c25f92a0ea09/Eu2epT1I01hBkOH5m0zYceIBFmiSyv7EKEbmYTAYsy6uMQ?e=gHj9Tj).
2. Place the downloaded dataset into the `datasets` directory at the root of the project.

## Training

Train the model using:

```bash
python model.py
```

## Testing

Test the trained model using:

```bash
python test.py
```

## Video Detection

To detect sign language from video input, run:

```bash
python detect.py
```
