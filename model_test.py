from ultralytics import YOLO

model = YOLO("model/v3/hand_sign_detector/weights/best.pt")

results = model.predict(
    source='datasets/asl-alphabet-v2/test/images',
    save=True,
    conf=0.5,
    iou=0.5,
    save_txt=True
)

