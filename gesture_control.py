import cv2
from cvzone.HandTrackingModule import HandDetector
import requests
import time

# CHANGE THIS to your ESP32 IP
ESP_URL = "http://172.20.10.3/cmd"  

detector = HandDetector(detectionCon=0.7, maxHands=1)
cap = cv2.VideoCapture(0)

def send(red, yellow, green):
    try:
        requests.post(
            ESP_URL,
            json={"red": red, "yellow": yellow, "green": green},
            timeout=0.2
        )
    except:
        pass  # ignore timeouts

while True:
    ok, img = cap.read()
    if not ok:
        time.sleep(0.05)
        continue

    hands, img = detector.findHands(img)
    red = yellow = green = 0

    if hands:
        fingers = detector.fingersUp(hands[0])
        count = sum(fingers)

        # Gesture rules:
        # Fist (0 fingers)      => Yellow
        # Open Hand (5 fingers) => Red
        # Only index up         => Green
        if count == 0:
            yellow = 1
        elif count == 5:
            red = 1
        elif fingers == [1,0,0,0,0]:
            green = 1

    send(red, yellow, green)

    cv2.imshow("Gesture Control", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()