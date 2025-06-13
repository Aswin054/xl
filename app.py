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
CORS(app)  # Enable CORS for all routes

# Configuration
app.config['UPLOAD_FOLDER'] = 'temp_uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB limit
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Initialize models
print("Loading YOLO model...")
model = YOLO("license_plate_detector.pt")
print("YOLO model loaded successfully")

print("Loading EasyOCR reader...")
reader = easyocr.Reader(['en'])
print("EasyOCR reader loaded successfully")

# Store detected plates
detected_plates = []

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

        # Debug: Save received frame for inspection
        debug_frame_path = os.path.join(app.config['UPLOAD_FOLDER'], f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        frame_file.save(debug_frame_path)
        
        # Read and verify image
        img_bytes = frame_file.read()
        if len(img_bytes) == 0:
            return jsonify({'success': False, 'error': 'Empty image data'}), 400

        frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'success': False, 'error': 'Invalid image data'}), 400

        # Process with YOLO
        results = model(frame)[0]
        detections = []
        
        print(f"YOLO detected {len(results.boxes)} objects")
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            print(f"Object detected with confidence: {conf:.2f}")
            
            if conf > 0.5:
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    print("Empty ROI - skipping")
                    continue
                
                # Save ROI for debugging
                roi_path = os.path.join(app.config['UPLOAD_FOLDER'], f"roi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                cv2.imwrite(roi_path, roi)
                
                # OCR with EasyOCR
                try:
                    ocr_results = reader.readtext(roi, detail=1)
                    print(f"OCR results: {ocr_results}")
                    
                    for (bbox, text, prob) in ocr_results:
                        clean_text = text.upper().replace(" ", "")
                        print(f"Raw OCR text: '{text}' -> Cleaned: '{clean_text}'")
                        
                        # More lenient validation for testing
                        if len(clean_text) >= 4:  # Minimum 4 characters
                            detection = {
                                'text': clean_text,
                                'confidence': float(conf),
                                'bbox': [x1, y1, x2, y2],
                                'ocr_confidence': float(prob)
                            }
                            detections.append(detection)
                            
                            # Check if this is a new plate
                            if not any(p['text'] == clean_text for p in detected_plates):
                                print(f"New plate detected: {clean_text}")
                                detected_plates.append({
                                    **detection,
                                    'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                })
                except Exception as ocr_error:
                    print(f"OCR Error: {str(ocr_error)}")
                    continue

        response = {
            'success': True,
            'detections': detections,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"Sending response: {response}")
        return jsonify(response)

    except Exception as e:
        print(f"Detection error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/upload', methods=['POST'])
def upload_video():
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'No video provided'}), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'}), 400

        # Save temp file
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{uuid.uuid4()}.mp4")
        video_file.save(temp_path)
        print(f"Video saved to: {temp_path}")
        
        # Process video
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            return jsonify({'success': False, 'error': 'Failed to open video'}), 400
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = max(1, int(fps / 2))  # Process ~2 FPS
        frame_count = 0
        
        print(f"Processing video at {fps:.1f} FPS, analyzing every {frame_skip} frames")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue
                
            # Process frame (same as detect_frame but without API response)
            results = model(frame)[0]
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                if conf > 0.5:
                    roi = frame[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue
                        
                    ocr_results = reader.readtext(roi)
                    for (_, text, prob) in ocr_results:
                        clean_text = text.upper().replace(" ", "")
                        if len(clean_text) >= 4:  # Lenient validation
                            if not any(p['text'] == clean_text for p in detected_plates):
                                detected_plates.append({
                                    'text': clean_text,
                                    'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    'detection_confidence': float(conf),
                                    'ocr_confidence': float(prob),
                                    'bbox': [x1, y1, x2, y2]
                                })
        
        cap.release()
        os.remove(temp_path)
        
        # Prepare results
        df = pd.DataFrame(detected_plates)
        output_filename = f"plates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        df.to_excel(output_path, index=False)
        
        return jsonify({
            'success': True,
            'message': 'Processing complete',
            'detected_plates': detected_plates,
            'download_url': f'/api/download/{output_filename}'
        })

    except Exception as e:
        print(f"Video processing error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/download/<filename>')
def download_file(filename):
    try:
        return send_file(
            os.path.join(app.config['OUTPUT_FOLDER'], filename),
            as_attachment=True,
            download_name=f"plate_detections_{datetime.now().strftime('%Y%m%d')}.xlsx"
        )
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 404

@app.route('/api/plates')
def get_plates():
    return jsonify({
        'success': True,
        'plates': detected_plates,
        'count': len(detected_plates)
    })

@app.route('/api/status')
def get_status():
    return jsonify({
        'success': True,
        'status': {
            'model_loaded': True,
            'plates_detected': len(detected_plates),
            'last_detection': detected_plates[-1]['time'] if detected_plates else None
        }
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)