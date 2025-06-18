from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import torch

app = Flask(__name__)
model = YOLO("best.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

cap = cv2.VideoCapture(0)
last_prediction = ""
counter = 0

def generate_frames():
    global last_prediction, counter

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model.predict(source=frame, conf=0.6, verbose=False)
        annotated_frame = results[0].plot()

        if results[0].boxes:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])

                # Stabilisasi prediksi
                if label == last_prediction:
                    counter += 1
                else:
                    counter = 1
                    last_prediction = label

                if counter >= 3:  # huruf akan tampil hanya jika stabil 3 frame berturut-turut
                    cv2.putText(
                        annotated_frame,
                        f"{label} ({conf:.2f})",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 255, 0),
                        2
                    )

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
