from ultralytics import YOLO

model = YOLO("model/v3/hand_sign_detector/weights/best.pt")

results = model.predict(
    source='datasets/sign-lang-v7/test/images',
    save=True,
    conf=0.5,
    iou=0.5,
    save_txt=True
)

