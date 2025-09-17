from ultralytics import YOLO

model = YOLO('yolo11n.pt')

results = model.train(
    data='datasets/sign-lang-v3/data.yaml',
    epochs=10,
    imgsz=640,
    name='hand_sign_detector_v11',
)

val_results = model.val()

model.save('hand_sign_detector_v11.pt')
