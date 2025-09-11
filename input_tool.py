import cv2
from ultralytics import YOLO
import time
import pyautogui
import win32gui
import win32con  # For window constants

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
            pyautogui.write(detected_text)  # Type the letter immediately
            word += detected_text  # Keep track for backspace functionality
            letter_ready = False
    else:
        last_letter = ''
        letter_ready = True
        letter_start_time = 0

    annotated_frame = results[0].plot()
    cv2.imshow("Sign Language Detector", annotated_frame)
    detector_window = win32gui.FindWindow(None, "Sign Language Detector")
    win32gui.SetWindowPos(detector_window, win32con.HWND_TOPMOST, 0, 0, 0, 0, 
                         win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)

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
    elif key == 13:  # Enter key
        pyautogui.write(word)
        print(f"Typed word: {word}")
    time.sleep(0.01)    

cap.release()
cv2.destroyAllWindows()
