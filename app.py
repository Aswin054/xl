import os
import cv2
import easyocr
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from datetime import datetime
from ultralytics import YOLO
import tempfile
import uuid
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = 'temp_uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Memory optimization settings
os.environ['YOLO_CONFIG_DIR'] = '/tmp'  # Use tmpfs for YOLO configs
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'  # Lazy load CUDA modules

# Global variables for models (initialized lazily)
model = None
reader = None
detected_plates = []

def get_yolo_model():
    global model
    if model is None:
        print("Loading YOLO model...")
        model = YOLO("license_plate_detector.pt", verbose=False)
        model.fuse()  # Optimize model
    return model

def get_ocr_reader():
    global reader
    if reader is None:
        print("Loading EasyOCR reader...")
        reader = easyocr.Reader(
            ['en'],
            gpu=False,  # Force CPU-only
            download_enabled=False,
            model_storage_directory='/tmp',
            quantize=True  # Use quantized model
        )
    return reader

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/detect', methods=['POST'])
def detect_frame():
    try:
        if 'frame' not in request.files:
            return jsonify({'success': False, 'error': 'No frame provided'}), 400
        
        frame_file = request.files['frame']
        if frame_file.filename == '':
            return jsonify({'success': False, 'error': 'Empty frame'}), 400

        # Process frame in memory without saving
        img_bytes = frame_file.read()
        frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'success': False, 'error': 'Invalid image data'}), 400

        # Get models (lazy loaded)
        yolo_model = get_yolo_model()
        ocr_reader = get_ocr_reader()

        # Process with YOLO
        results = yolo_model(frame, imgsz=320)[0]  # Smaller image size
        detections = []
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            if conf > 0.5:
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                
                # OCR with EasyOCR
                try:
                    ocr_results = ocr_reader.readtext(
                        roi,
                        batch_size=1,  # Process one at a time
                        decoder='beamsearch',
                        beamWidth=3
                    )
                    
                    for (bbox, text, prob) in ocr_results:
                        clean_text = text.upper().replace(" ", "")
                        if len(clean_text) >= 4:
                            detections.append({
                                'text': clean_text,
                                'confidence': float(conf),
                                'bbox': [x1, y1, x2, y2],
                                'ocr_confidence': float(prob)
                            })
                            
                            # Store unique plates
                            if not any(p['text'] == clean_text for p in detected_plates):
                                detected_plates.append({
                                    'text': clean_text,
                                    'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    'detection_confidence': float(conf),
                                    'ocr_confidence': float(prob)
                                })
                except Exception as ocr_error:
                    print(f"OCR Error: {ocr_error}")
                    continue

        return jsonify({
            'success': True,
            'detections': detections,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        print(f"Detection error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_video():
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'No video provided'}), 400
        
        video_file = request.files['video']
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{uuid.uuid4()}.mp4")
        video_file.save(temp_path)
        
        # Get models (lazy loaded)
        yolo_model = get_yolo_model()
        ocr_reader = get_ocr_reader()

        cap = cv2.VideoCapture(temp_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = max(1, int(fps / 2))  # Process ~2 FPS
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % frame_skip != 0:
                continue
                
            # Process frame
            results = yolo_model(frame, imgsz=320)[0]
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                if conf > 0.5:
                    roi = frame[y1:y2, x1:x2]
                    if roi.size > 0:
                        ocr_results = ocr_reader.readtext(roi, batch_size=1)
                        for (_, text, prob) in ocr_results:
                            clean_text = text.upper().replace(" ", "")
                            if len(clean_text) >= 4 and not any(p['text'] == clean_text for p in detected_plates):
                                detected_plates.append({
                                    'text': clean_text,
                                    'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    'detection_confidence': float(conf),
                                    'ocr_confidence': float(prob),
                                    'bbox': [x1, y1, x2, y2]
                                })
        
        cap.release()
        os.remove(temp_path)
        
        # Save results
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"plates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
        pd.DataFrame(detected_plates).to_excel(output_path, index=False)
        
        return jsonify({
            'success': True,
            'message': 'Processing complete',
            'download_url': f'/api/download/{os.path.basename(output_path)}'
        })

    except Exception as e:
        print(f"Video processing error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ... (keep the remaining routes unchanged)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)