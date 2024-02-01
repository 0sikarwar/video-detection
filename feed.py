from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

# Adjust these values according to your needs
frame_width = 640
frame_height = 480
frame_rate = 20  # Frames per second
quality = 70  # JPEG quality (1-100)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
cap.set(cv2.CAP_PROP_FPS, frame_rate)

def generate_frames():
    while True:
        try:
            success, frame = cap.read()
            if not success or frame is None or frame.size == 0:
                print("Failed to grab frame")
                continue

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"Error generating frame: {str(e)}")
            continue  # Auto-restart logic

@app.route('/')
def index():
    return render_template('feed.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True, threaded=True)
