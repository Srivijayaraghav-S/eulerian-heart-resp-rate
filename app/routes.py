from flask import render_template, jsonify, request, Response, send_file
import threading
from app import app
import fall, collab_cam, eulerian_main
from shared_state import final_heart_rate, recording_complete_flag, fall_detected_flag

process_threads = {}
fall_detected = False
final_hr_data = {}  # To store HR info (e.g., final reading or plot filename)

def run_in_thread(target, name):
    def wrapper():
        try:
            target()
        except Exception as e:
            print(f"Error in thread {name}: {e}")
    thread = threading.Thread(target=wrapper, name=name)
    thread.daemon = True
    thread.start()
    process_threads[name] = thread

def set_fall_detected():
    from shared_state import fall_detected_flag
    fall_detected_flag["value"] = True
    
    

@app.route('/get_final_heart_rate')
def get_final_heart_rate():
    print("Current heart rate:", final_heart_rate["value"])  # Debug log
    if final_heart_rate["value"] is not None:
        return jsonify({"status": "done", "heart_rate": f"{final_heart_rate['value']:.1f}"})
    return jsonify({"status": "processing"})


@app.route('/')
def index():
    global fall_detected_flag, recording_complete_flag, fall_detected, final_hr_data
    from shared_state import final_heart_rate

    # Reset all runtime flags and values
    fall_detected_flag["value"] = False
    recording_complete_flag["value"] = False
    final_heart_rate["value"] = None
    final_hr_data = {}
    fall_detected = False

    # Optionally delete old plot and log files
    import os
    try:
        if os.path.exists("plots/hr_monitoring.png"):
            os.remove("plots/hr_monitoring.png")
        if os.path.exists("logs/final_heart_rate.json"):
            os.remove("logs/final_heart_rate.json")
    except Exception as e:
        print(f"Cleanup error: {e}")
    
    if os.path.exists("face_tracking_output_new.mov"):
        os.remove("face_tracking_output_new.mov")
    
    return render_template('index.html')


@app.route('/fall_detection')
def fall_detection():
    return Response(
        fall.fall_detection_stream(set_fall_detected_callback=set_fall_detected),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/face_tracking')
def face_tracking():
    # Return the combined stream (face tracking recording followed by Eulerian processing)
    # In our collab_cam_stream() generator, after finishing face tracking,
    # we call eulerian_main processing.
    return Response(
        collab_cam.collab_cam_stream(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/start_eulerian', methods=['POST'])
def start_eulerian():
    # For example, start Eulerian processing in a separate thread.
    run_in_thread(eulerian_main.eulerian_main_stream, "heart_rate_monitoring")
    return jsonify({"status": "Eulerian processing started"}), 200

@app.route('/fall_status')
def fall_status():
    return jsonify({'fall_detected': fall_detected})

@app.route('/recording_complete')
def recording_complete():
    global recording_complete_flag
    recording_complete_flag = True
    return jsonify({'recording_complete': True})

@app.route('/recording_status')
def recording_status():
    return jsonify({'recording_complete': recording_complete_flag})

@app.route('/status_check')
def status_check():
    print("Status check: ", fall_detected_flag, recording_complete_flag)
    return jsonify({
        'fall_detected': fall_detected_flag["value"],
        'recording_complete': recording_complete_flag["value"],
    })

