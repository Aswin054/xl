import cv2
import easyocr
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
import os

# Load YOLO model
model = YOLO("license_plate_detector(1) (1).pt")

# OCR reader
reader = easyocr.Reader(['en'])

# Create output folder
if not os.path.exists("output"):
    os.mkdir("output")
excel_path = "output/log.xlsx"
plate_log = []

# Set video path or webcam
use_webcam = False  # ✅ Set to True for live webcam feed
video_path = r"C:\Users\Lenova\Desktop\xl\WhatsApp Video 2025-03-19 at 13.14.32_84ae950f.mp4"

cap = cv2.VideoCapture(0 if use_webcam else video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        if conf > 0.5:
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            result = reader.readtext(roi)
            for (_, text, prob) in result:
                text = text.upper().replace(" ", "")
                if text.startswith("TN") and 6 <= len(text) <= 12:
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    plate_log.append({'Plate': text, 'Time': timestamp})
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("License Plate Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save to Excel
df = pd.DataFrame(plate_log)
if not df.empty:
    df.drop_duplicates(subset='Plate', inplace=True)
    if os.path.exists(excel_path):
        os.remove(excel_path)
    df.to_excel(excel_path, index=False)
    print(f"✅ Log saved to: {excel_path}")
else:
    print("⚠️ No TN plates detected.")
