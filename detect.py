import cv2
import numpy as np
from ultralytics import YOLO
import time

model = YOLO("model/v3/hand_sign_detector/weights/best.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    results = model.predict(frame, imgsz=640, conf=0.5)

    # Extract detected class names
    detected_classes = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        class_name = results[0].names[cls_id]
        detected_classes.append(class_name)
    # Only show the first detected class (if any)
    detected_text = detected_classes[0] if detected_classes else ''

    text_img = 255 * np.ones((100, 400, 3), dtype=np.uint8)
    cv2.putText(text_img, detected_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    cv2.imshow("Detected Classes", text_img)

    annotated_frame = results[0].plot()

    cv2.imshow("Sign Language Detector", annotated_frame)

    if cv2.waitKey(1) == 27:  # Escape key
        break
    time.sleep(0.1)

cap.release()
cv2.destroyAllWindows()
