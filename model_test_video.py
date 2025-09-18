import cv2
import numpy as np
from ultralytics import YOLO

# Load your trained YOLO model
model = YOLO("model/v3/hand_sign_detector/weights/best.pt")

# Replace 'video.mp4' with the path to your video file
cap = cv2.VideoCapture("videos/sample_1.mp4")

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get the original video FPS
fps = cap.get(cv2.CAP_PROP_FPS)
frame_delay = int(1000 / fps * 5)  # Slow down to 1/5 speed

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Make prediction
    results = model.predict(frame, imgsz=640, conf=0.5)

    # Get detected classes
    detected_classes = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        class_name = results[0].names[cls_id]
        detected_classes.append(class_name)
    detected_text = detected_classes[0] if detected_classes else ''

    # Display detected text
    text_img = 255 * np.ones((100, 400, 3), dtype=np.uint8)
    cv2.putText(text_img, detected_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    cv2.imshow("Detected Classes", text_img)

    # Display annotated frame
    annotated_frame = results[0].plot()
    cv2.imshow("Sign Language Detector", annotated_frame)

    # Wait time adjusted for 1/5 speed
    if cv2.waitKey(frame_delay) & 0xFF == 27:  # Escape key to exit
        break

cap.release()
cv2.destroyAllWindows()
