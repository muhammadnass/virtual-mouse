import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# ============================= CONFIGURATION =============================
wCam, hCam       = 640, 480   # Camera resolution
frameR           = 100        # Frame reduction margin (active zone border)
smoothening      = 6          # Smoothing factor (higher = smoother but laggier)

# Gesture distance thresholds (pixels in camera space)
click_threshold        = 35   # Index + Middle close  → left click
double_click_threshold = 45   # Index + Middle + Ring close → double click
right_click_threshold  = 80   # Index + Middle spread → right click
drag_engage_threshold  = 50   # Thumb + Index close   → start drag
scroll_threshold       = 30   # Index + Pinky close   → scroll mode active

# Per-gesture cooldowns (seconds)
COOLDOWN = {
    "left_click":    0.40,
    "double_click":  0.60,
    "right_click":   0.60,
    "scroll":        0.05,
}

SCROLL_SPEED = 3              # Lines scrolled per trigger

# ============================= HELPERS ===================================

def dist(p1, p2):
    """Euclidean distance between two (x, y) landmark points."""
    return np.hypot(p2[0] - p1[0], p2[1] - p1[1])


def get_fingers(lm, handedness="Right"):
    """
    Returns a 5-element list [thumb, index, middle, ring, pinky].
    1 = extended, 0 = folded.
    Handles both Right and Left hands correctly.
    """
    fingers = []

    # Thumb: compare tip (4) to IP joint (3) along x-axis,
    # direction flipped for left hand.
    if handedness == "Right":
        fingers.append(1 if lm[4][1] < lm[3][1] else 0)
    else:
        fingers.append(1 if lm[4][1] > lm[3][1] else 0)

    # Index, Middle, Ring, Pinky: tip y < pip y  → extended
    for tip in [8, 12, 16, 20]:
        fingers.append(1 if lm[tip][2] < lm[tip - 2][2] else 0)

    return fingers


def open_camera():
    """Try multiple backends; return the first that works."""
    backends = [cv2.CAP_V4L2, cv2.CAP_GSTREAMER, cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(0, backend)
        if cap.isOpened():
            return cap
    raise RuntimeError("No camera backend could open device 0.")


def draw_hud(img, fps, paused, gesture_label, fingers):
    """Overlay FPS, pause state, active gesture, and finger state."""
    h, w = img.shape[:2]

    # Semi-transparent sidebar
    overlay = img.copy()
    cv2.rectangle(overlay, (w - 160, 0), (w, 180), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    color = (0, 255, 0) if not paused else (0, 60, 255)
    status = "ACTIVE" if not paused else "PAUSED"
    cv2.putText(img, status, (w - 150, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(img, f"FPS: {fps:5.1f}", (w - 150, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Finger indicators
    labels = ["T", "I", "M", "R", "P"]
    for i, (label, state) in enumerate(zip(labels, fingers)):
        c = (0, 230, 80) if state else (80, 80, 80)
        cv2.circle(img, (w - 145 + i * 26, 90), 10, c, -1)
        cv2.putText(img, label, (w - 152 + i * 26, 94),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)

    if gesture_label:
        cv2.putText(img, gesture_label, (w - 155, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 220, 0), 2)

    # Controls reminder
    cv2.putText(img, "P=Pause  Q=Quit", (w - 155, 168),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)


# ============================= INITIALIZATION ============================
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.65,
)

cap = open_camera()
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  wCam)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FPS, 60)

wScr, hScr = pyautogui.size()

# Smoothing state
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0

# Per-gesture last-fired timestamps
last_action = {k: 0.0 for k in COOLDOWN}

# Drag state (tracked across frames, not re-fired every frame)
is_dragging = False

# App state
paused        = False
gesture_label = ""
fps           = 0.0
prev_frame_t  = time.time()

print("=" * 55)
print("  Enhanced Virtual Mouse  |  Ubuntu / PyCharm")
print("=" * 55)
print("Gestures:")
print("  Index only              → Move cursor")
print("  Index + Middle close    → Left click")
print("  Index + Mid + Ring      → Double click")
print("  Index + Middle spread   → Right click")
print("  Thumb + Index close     → Drag / hold")
print("  Index + Pinky up        → Scroll (move hand up/down)")
print("Keyboard:")
print("  P  →  Pause / Resume")
print("  Q  →  Quit")
print("=" * 55)

# ============================= MAIN LOOP =================================
while True:
    success, img = cap.read()
    if not success:
        print("Warning: dropped frame — retrying...")
        continue

    # FPS calculation
    now       = time.time()
    fps       = 0.9 * fps + 0.1 * (1.0 / max(now - prev_frame_t, 1e-6))
    prev_frame_t = now

    img = cv2.flip(img, 1)

    # Active zone rectangle
    cv2.rectangle(img, (frameR, frameR),
                  (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

    lmList      = []
    handedness  = "Right"
    fingers     = [0, 0, 0, 0, 0]
    gesture_label = ""

    if not paused:
        imgRGB  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks and results.multi_handedness:
            myHand      = results.multi_hand_landmarks[0]
            hand_label  = results.multi_handedness[0].classification[0].label
            handedness  = hand_label   # "Left" or "Right" (already mirrored by flip)

            h, w, _ = img.shape
            for id, lm in enumerate(myHand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if id == 8:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

            mp_draw.draw_landmarks(img, myHand, mp_hands.HAND_CONNECTIONS)

        if len(lmList) == 21:
            fingers = get_fingers(lmList, handedness)

            # Key landmark shortcuts
            ix, iy = lmList[8][1],  lmList[8][2]   # Index tip
            mx, my = lmList[12][1], lmList[12][2]  # Middle tip
            tx, ty = lmList[4][1],  lmList[4][2]   # Thumb tip
            px, py = lmList[20][1], lmList[20][2]  # Pinky tip
            t       = time.time()

            # ============================================================
            # SCROLL MODE: Index + Pinky up, others down
            # Move hand up → scroll up; down → scroll down.
            # ============================================================
            if fingers[1] == 1 and fingers[4] == 1 and fingers[2] == 0 and fingers[3] == 0:
                gesture_label = "SCROLL"
                # Use index tip y mapped to scroll direction
                mid_y = hCam // 2
                if t - last_action["scroll"] > COOLDOWN["scroll"]:
                    direction = -SCROLL_SPEED if iy < mid_y else SCROLL_SPEED
                    pyautogui.scroll(direction)
                    last_action["scroll"] = t
                cv2.circle(img, (ix, iy), 15, (255, 165, 0), cv2.FILLED)

            # ============================================================
            # MOVE MODE: Only index finger extended
            # ============================================================
            elif fingers == [0, 1, 0, 0, 0]:
                x3 = np.interp(ix, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(iy, (frameR, hCam - frameR), (0, hScr))

                curr_x = prev_x + (x3 - prev_x) / smoothening
                curr_y = prev_y + (y3 - prev_y) / smoothening
                pyautogui.moveTo(int(curr_x), int(curr_y))

                cv2.circle(img, (ix, iy), 15, (0, 255, 0), cv2.FILLED)
                prev_x, prev_y = curr_x, curr_y
                gesture_label = "MOVE"

            # ============================================================
            # DOUBLE CLICK: Index + Middle + Ring up
            # ============================================================
            elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 0:
                d = dist((ix, iy), (mx, my))
                cv2.line(img, (ix, iy), (mx, my), (0, 255, 255), 2)
                if d < double_click_threshold and t - last_action["double_click"] > COOLDOWN["double_click"]:
                    pyautogui.doubleClick()
                    last_action["double_click"] = t
                    gesture_label = "DOUBLE CLICK"
                    cv2.circle(img, (ix, iy), 15, (255, 255, 0), cv2.FILLED)

            # ============================================================
            # LEFT CLICK: Index + Middle up only, fingers close together
            # ============================================================
            elif fingers[1] == 1 and fingers[2] == 1 and sum(fingers) == 2:
                d = dist((ix, iy), (mx, my))
                cv2.line(img, (ix, iy), (mx, my), (0, 255, 255), 2)
                if d < click_threshold and t - last_action["left_click"] > COOLDOWN["left_click"]:
                    pyautogui.click()
                    last_action["left_click"] = t
                    gesture_label = "LEFT CLICK"
                    cv2.circle(img, (ix, iy), 15, (0, 0, 255), cv2.FILLED)
                elif d >= right_click_threshold and t - last_action["right_click"] > COOLDOWN["right_click"]:
                    # ======================================================
                    # RIGHT CLICK: same fingers but spread wide
                    # ======================================================
                    pyautogui.rightClick()
                    last_action["right_click"] = t
                    gesture_label = "RIGHT CLICK"
                    cv2.circle(img, (ix, iy), 15, (255, 0, 0), cv2.FILLED)
                    cv2.line(img, (ix, iy), (mx, my), (255, 0, 0), 2)

            # ============================================================
            # DRAG & DROP: Thumb + Index up
            # State-tracked: mouseDown only on transition, not every frame.
            # ============================================================
            elif fingers[0] == 1 and fingers[1] == 1 and sum(fingers) == 2:
                d_ti = dist((tx, ty), (ix, iy))
                cv2.line(img, (tx, ty), (ix, iy), (0, 255, 0), 2)

                if d_ti < drag_engage_threshold:
                    if not is_dragging:
                        pyautogui.mouseDown()
                        is_dragging = True
                    gesture_label = "DRAG"
                    cv2.circle(img, (ix, iy), 15, (0, 200, 0), cv2.FILLED)
                else:
                    if is_dragging:
                        pyautogui.mouseUp()
                        is_dragging = False
                    gesture_label = "RELEASE"

            else:
                # No recognised gesture — release drag if still held
                if is_dragging:
                    pyautogui.mouseUp()
                    is_dragging = False

    # HUD overlay
    draw_hud(img, fps, paused, gesture_label, fingers)

    cv2.imshow("Virtual Mouse Enhanced  |  P=Pause  Q=Quit", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p') or key == ord('P'):
        paused = not paused
        if paused and is_dragging:          # Safety: release drag on pause
            pyautogui.mouseUp()
            is_dragging = False
        print("Paused." if paused else "Resumed.")

# ============================= CLEANUP ===================================
if is_dragging:
    pyautogui.mouseUp()

cap.release()
cv2.destroyAllWindows()
print("Virtual Mouse closed.")
