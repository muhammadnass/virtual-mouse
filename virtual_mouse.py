import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# ============================= CONFIGURATION =============================
wCam, hCam = 640, 480           # Camera resolution
frameR = 120                    # Frame reduction (margin for hand movement)
smoothening = 5                 # Smoothing factor (lower = faster, higher = smoother)
click_threshold = 35            # Distance for left click (pixels)
double_click_threshold = 45     # Distance for double click
right_click_threshold = 80      # Distance for right click (spread fingers)

# ============================= INITIALIZATION =============================
pyautogui.FAILSAFE = False      # Disable failsafe to avoid accidental exits
pyautogui.PAUSE = 0             # Reduce delay between actions

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # Prefer V4L2 backend for Linux stability

if not cap.isOpened():
    print("Error: Could not open webcam with V4L2.")
    cap = cv2.VideoCapture(0)  # Fallback to default
    if not cap.isOpened():
        print("Failed to open camera with any backend. Exiting...")
        exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

# Get screen resolution
wScr, hScr = pyautogui.size()

# Variables for smoothing
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0

# Flag to prevent multiple clicks in short time
last_click_time = 0

print("Virtual Mouse started. Press 'q' to quit.")
print("Gestures:")
print("  • Only index finger up → Move cursor")
print("  • Index + Middle fingers close → Left click")
print("  • Index + Middle + Ring fingers close → Double click")
print("  • Index + Middle fingers spread wide → Right click")
print("  • Thumb + Index close → Drag (hold), separate → Release")

# ============================= MAIN LOOP =============================
while True:
    success, img = cap.read()
    if not success:
        print("Warning: Failed to grab frame. Retrying...")
        continue

    # Mirror the image for natural movement
    img = cv2.flip(img, 1)

    # Process hand landmarks
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # Draw movement area rectangle
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                  (255, 0, 255), 2)

    lmList = []

    if results.multi_hand_landmarks:
        myHand = results.multi_hand_landmarks[0]
        h, w, _ = img.shape

        # Extract landmarks
        for id, lm in enumerate(myHand.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append([id, cx, cy])

            # Highlight index fingertip
            if id == 8:
                cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        mp_draw.draw_landmarks(img, myHand, mp_hands.HAND_CONNECTIONS)

    if len(lmList) >= 21:  # Ensure we have all major landmarks
        # ==================== FINGER DETECTION ====================
        fingers = []

        # Thumb (tip x > IP joint x for typical right-hand gesture)
        if lmList[4][1] > lmList[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Index, Middle, Ring, Pinky
        tip_ids = [8, 12, 16, 20]
        for tip in tip_ids:
            if lmList[tip][2] < lmList[tip - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # Get key points
        x1, y1 = lmList[8][1:]   # Index tip
        x2, y2 = lmList[12][1:]  # Middle tip

        # ==================== MOVE MODE ====================
        if fingers == [0, 1, 0, 0, 0]:  # Only index finger up
            # Map camera coords to screen
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            # Smooth movement
            curr_x = prev_x + (x3 - prev_x) / smoothening
            curr_y = prev_y + (y3 - prev_y) / smoothening

            pyautogui.moveTo(int(curr_x), int(curr_y))

            cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
            prev_x, prev_y = curr_x, curr_y

        # ==================== GESTURES ====================
        current_time = time.time()

        # Left click: Index + Middle close
        if fingers[1] == 1 and fingers[2] == 1 and sum(fingers) == 2:
            length = np.hypot(x2 - x1, y2 - y1)
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 3)

            if length < click_threshold and current_time - last_click_time > 0.4:
                cv2.circle(img, (x1, y1), 15, (0, 0, 255), cv2.FILLED)
                pyautogui.click()
                last_click_time = current_time
                cv2.putText(img, "LEFT CLICK", (x1 - 80, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Double click: Index + Middle + Ring close
        elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
            length = np.hypot(x2 - x1, y2 - y1)
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 3)

            if length < double_click_threshold and current_time - last_click_time > 0.6:
                cv2.circle(img, (x1, y1), 15, (255, 255, 0), cv2.FILLED)
                pyautogui.doubleClick()
                last_click_time = current_time
                cv2.putText(img, "DOUBLE CLICK", (x1 - 100, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Right click: Index + Middle spread wide
        elif fingers[1] == 1 and fingers[2] == 1:
            length = np.hypot(x2 - x1, y2 - y1)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

            if length > right_click_threshold and current_time - last_click_time > 0.6:
                cv2.circle(img, (x1, y1), 15, (255, 0, 0), cv2.FILLED)
                pyautogui.rightClick()
                last_click_time = current_time
                cv2.putText(img, "RIGHT CLICK", (x1 - 90, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Drag & Drop: Thumb + Index close (hold), separate (release)
        elif fingers[0] == 1 and fingers[1] == 1 and sum(fingers) == 2:
            length_thumb_index = np.hypot(lmList[4][1] - x1, lmList[4][2] - y1)
            cv2.line(img, (lmList[4][1], lmList[4][2]), (x1, y1), (0, 255, 0), 3)

            if length_thumb_index < 50:
                pyautogui.mouseDown()
                cv2.putText(img, "DRAG", (x1 - 40, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                pyautogui.mouseUp()
                cv2.putText(img, "RELEASE", (x1 - 60, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Virtual Mouse - Press 'q' to quit", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ============================= CLEANUP =============================
cap.release()
cv2.destroyAllWindows()
print("Virtual Mouse closed.")