from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(
    data='datasets/sign-lang-v7/data.yaml',
    epochs=10,
    imgsz=640,
    name='hand_sign_detector',
)

val_results = model.val()

model.save('hand_sign_detector.pt')
