import cv2
import time
import threading
import os
from flask import Flask, render_template, Response, jsonify, request
from werkzeug.utils import secure_filename
from ensemble_detector import YoloFastSAMEnsemble
from ensemble_config import CONFIG

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

current_system_state = {
    "risk_level": "SAFE",
    "fps": 0,
    "critical_count": 0,
    "caution_count": 0,
    "detection_count": 0,
    "was_enhanced": False
}
lock = threading.Lock()

video_source = 0 

CONFIG["verbose"] = False
ensemble = YoloFastSAMEnsemble(CONFIG)

def generate_frames():
    """
    Video streaming generator. 
    Handles both Webcam (0) and Video Files (path).
    """
    global video_source, current_system_state
    
    cap = cv2.VideoCapture(video_source)
    
    while True:
        success, frame = cap.read()
        
        if not success:
            if isinstance(video_source, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        results = ensemble.process_frame(frame)
        annotated_frame = results["annotated_frame"]
        
        stats = results.get("system_state", {})
        fps = results.get("fps", 0)
        
        with lock:
            current_system_state = {
                "risk_level": stats.get("max_risk_level", "SAFE"),
                "fps": round(fps, 1),
                "critical_count": stats.get("critical_count", 0),
                "caution_count": stats.get("caution_count", 0),
                "detection_count": len(results["fused"]),
                "was_enhanced": results.get("was_enhanced", False)
            }

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/drive')
def drive():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def get_stats():
    with lock:
        return jsonify(current_system_state)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle video file uploads and switch source."""
    global video_source
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        video_source = filepath
        return jsonify({'success': True, 'type': 'video', 'name': filename})

@app.route('/reset_source')
def reset_source():
    """Switch back to Webcam."""
    global video_source
    video_source = 0
    return jsonify({'success': True, 'type': 'webcam'})

if __name__ == "__main__":
    print("Starting ADAS Web Interface...")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)