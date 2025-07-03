import cv2
from deepface import DeepFace

# --------------------------- config ---------------------------------
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FONT         = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE   = 0.9
THICKNESS    = 2
BOX_COLOR    = (255, 0, 0)      # blue rectangle
TEXT_COLOR   = (0, 255, 0)      # green label
MIN_SIZE     = 50               # ignore faces smaller than this (px)
# --------------------------------------------------------------------

# Load Haar Cascade once
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# Open webcam
cap = cv2.VideoCapture(0)             # change index if needed
if not cap.isOpened():
    raise IOError("Could not open webcam")

print("[INFO] Webcam OK.  Press  q  to quit.")

while True:
    ok, frame = cap.read()
    if not ok:
        print("[WARN] Empty frame – skipping")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(MIN_SIZE, MIN_SIZE)
    )

    for (x, y, w, h) in faces:
        # draw face rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_COLOR, 2)

        # skip ROIs that are still too small for DeepFace
        if w < MIN_SIZE or h < MIN_SIZE:
            cv2.putText(
                frame,
                "Face too small",
                (x, y - 10),
                FONT,
                FONT_SCALE,
                (0, 0, 255),
                THICKNESS,
            )
            continue

        face_roi = frame[y : y + h, x : x + w]

        try:
            # DeepFace needs BGR (OpenCV default) and prefers silent=True
            result = DeepFace.analyze(
                face_roi,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
            )

            # ➤ CHANGES: result is now a dict.
            # Wrap into list only if DeepFace returned a list
            if isinstance(result, list):
                result = result[0]

            dominant = result["dominant_emotion"]
            score    = result["emotion"][dominant] * 100  # to %
            label    = f"{dominant} ({score:.1f}%)"

            # draw label
            cv2.putText(
                frame,
                label,
                (x, max(y - 10, 20)),  # keep text visible when y is small
                FONT,
                FONT_SCALE,
                TEXT_COLOR,
                THICKNESS,
            )

        except Exception as e:
            # DeepFace sometimes fails on blurred / side faces → just show hint
            cv2.putText(
                frame,
                "Analyzing...",
                (x, max(y - 10, 20)),
                FONT,
                FONT_SCALE,
                (0, 165, 255),
                THICKNESS,
            )

    cv2.imshow("Real‑Time Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
