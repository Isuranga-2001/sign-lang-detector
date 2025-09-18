from ultralytics import YOLO

model = YOLO("model/v5/hand_sign_detector_v11/weights/best.pt")

results = model.predict(
    source='datasets/sign-lang-v3/test/images',
    save=True,
    conf=0.5,
    iou=0.5,
    save_txt=True
)

