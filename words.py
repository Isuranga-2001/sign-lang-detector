import cv2
import numpy as np
from ultralytics import YOLO
import time

model = YOLO("model/v2/hand_sign_detector/weights/best.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access webcam.")
    exit()

word = ''
last_letter = ''
letter_ready = True
letter_start_time = 0
min_detection_time = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    results = model.predict(frame, imgsz=640, conf=0.5)

    detected_classes = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        class_name = results[0].names[cls_id]
        detected_classes.append(class_name)
    detected_text = detected_classes[0] if detected_classes else ''

    current_time = time.time()
    if detected_text:
        if detected_text != last_letter:
            letter_start_time = current_time
            last_letter = detected_text
        elif current_time - letter_start_time >= min_detection_time and letter_ready:
            word += detected_text
            letter_ready = False
    else:
        last_letter = ''
        letter_ready = True
        letter_start_time = 0

    word_img = 255 * np.ones((100, 400, 3), dtype=np.uint8)
    cv2.putText(word_img, word, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    cv2.imshow("Current Word", word_img)

    annotated_frame = results[0].plot()
    cv2.imshow("Sign Language Detector", annotated_frame)

    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break
    elif key == ord('r'):
        word = ''
        last_letter = ''
        letter_ready = True
    elif key == 8:  # Backspace key
        word = word[:-1]
        last_letter = ''
        letter_ready = True
    elif key == 32:  # Spacebar key
        word += ' '
        last_letter = ''
        letter_ready = True
    time.sleep(0.01)    

cap.release()
cv2.destroyAllWindows()
