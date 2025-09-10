import cv2
from ultralytics import YOLO

# Load your trained YOLO model (change path if needed)
model = YOLO("model/v3/hand_sign_detector/weights/best.pt")

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access webcam.")
    exit()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Run YOLO detection on the frame
    results = model.predict(frame, imgsz=640, conf=0.5)

    # Get annotated frame (YOLO automatically draws boxes + labels)
    annotated_frame = results[0].plot()

    # Show the frame
    cv2.imshow("Hand Pose Detector", annotated_frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
