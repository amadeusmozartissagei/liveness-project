import math
import time
import cv2
import cvzone
from ultralytics import YOLO
from flask import Flask, Response

# Inisialisasi Flask app
app = Flask(__name__)

# Inisialisasi model YOLO
model = YOLO("../models/best_204.pt")
confidence = 0.6
classNames = ["fake", "real"]

# Membuka kamera
cap = cv2.VideoCapture(0)  # Untuk Webcam
cap.set(3, 360)  # Lebar (9 bagian)
cap.set(4, 640)  # Tinggi (16 bagian)

# Menghitung FPS
prev_frame_time = 0
new_frame_time = 0

# Variabel untuk tracking deteksi wajah
face_detection_start = None
current_detection = None
verification_status = None
show_notification = False
notification_start = None
is_resetting = False
reset_countdown = 0
RESET_COUNTDOWN_DURATION = 3  # Durasi countdown reset dalam detik
NOTIFICATION_DURATION = 3  # Durasi notifikasi dalam detik
VERIFICATION_THRESHOLD = 10  # Waktu yang dibutuhkan untuk verifikasi dalam detik


def reset_verification():
    global face_detection_start, current_detection, verification_status, show_notification, is_resetting, reset_countdown
    face_detection_start = None
    current_detection = None
    verification_status = None
    show_notification = False
    is_resetting = True
    reset_countdown = RESET_COUNTDOWN_DURATION


def start_new_verification():
    global is_resetting, reset_countdown
    is_resetting = False
    reset_countdown = 0


# Fungsi untuk menangani stream video
def generate_frames():
    global prev_frame_time, face_detection_start, current_detection, verification_status
    global show_notification, notification_start, is_resetting, reset_countdown

    while True:
        new_frame_time = time.time()
        success, img = cap.read()

        if not success:
            break

        # Handle reset countdown
        if is_resetting:
            if reset_countdown > 0:
                cvzone.putTextRect(img, f"Starting new scan in: {int(reset_countdown)}s",
                                   (int(img.shape[1] / 2) - 150, int(img.shape[0] / 2)),
                                   scale=2, thickness=2, colorR=(255, 165, 0))
                cvzone.putTextRect(img, "Please position your face",
                                   (int(img.shape[1] / 2) - 120, int(img.shape[0] / 2) + 40),
                                   scale=1, thickness=1)
                reset_countdown -= 1 / 30  # Assuming 30 FPS
            else:
                start_new_verification()

            # Convert frame to JPEG during reset countdown
            ret, buffer = cv2.imencode('.jpg', img)
            if not ret:
                break
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            continue

        results = model(img, stream=True, verbose=False)
        current_frame_detection = None

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])

                if conf > confidence:
                    current_frame_detection = classNames[cls]

                    if classNames[cls] == 'real':
                        color = (0, 255, 0)  # Green for 'real'
                    else:
                        color = (0, 0, 255)  # Red for 'fake'

                    # Draw the bounding box and label
                    cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)
                    cvzone.putTextRect(img, f'{classNames[cls].upper()} {int(conf * 100)}%',
                                       (max(0, x1), max(35, y1)), scale=2, thickness=4, colorR=color,
                                       colorB=color)

        # Logic verifikasi wajah
        if current_frame_detection:
            if current_detection != current_frame_detection:
                current_detection = current_frame_detection
                face_detection_start = time.time()
            elif face_detection_start and not verification_status:
                elapsed_time = time.time() - face_detection_start

                # Tampilkan countdown verifikasi
                countdown = max(0, VERIFICATION_THRESHOLD - int(elapsed_time))
                if countdown > 0:
                    cvzone.putTextRect(img, f"Verifying: {countdown}s",
                                       (10, img.shape[0] - 20), scale=2, thickness=2)
                    # Tambahkan instruksi untuk pengguna
                    cvzone.putTextRect(img, "Keep your face steady",
                                       (10, img.shape[0] - 60), scale=1, thickness=1)

                if elapsed_time >= VERIFICATION_THRESHOLD:
                    verification_status = "SUCCESS" if current_detection == "real" else "FAILED"
                    show_notification = True
                    notification_start = time.time()
        else:
            if face_detection_start and not verification_status:
                # Reset jika tidak ada wajah terdeteksi
                cvzone.putTextRect(img, "No face detected! Please position your face.",
                                   (10, img.shape[0] - 20), scale=1, thickness=1)
                if time.time() - face_detection_start > 2:  # Reset after 2 seconds of no detection
                    reset_verification()

        # Tampilkan notifikasi hasil
        if show_notification:
            notification_elapsed = time.time() - notification_start
            if notification_elapsed < NOTIFICATION_DURATION:
                if verification_status == "SUCCESS":
                    cvzone.putTextRect(img, "Verification Successful!",
                                       (int(img.shape[1] / 2) - 100, int(img.shape[0] / 2)),
                                       scale=2, thickness=2, colorR=(0, 255, 0))
                else:
                    cvzone.putTextRect(img, "Verification Failed!",
                                       (int(img.shape[1] / 2) - 100, int(img.shape[0] / 2)),
                                       scale=2, thickness=2, colorR=(0, 0, 255))
                cvzone.putTextRect(img, "Press 'R' to scan again",
                                   (int(img.shape[1] / 2) - 100, int(img.shape[0] / 2) + 40),
                                   scale=1, thickness=1)
            else:
                reset_verification()  # Auto-reset after notification duration

        # Menghitung FPS
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        print(fps)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r') or key == ord('R'):
            reset_verification()

        # Mengubah frame menjadi format JPEG
        ret, buffer = cv2.imencode('.jpg', img)
        if not ret:
            break
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


# Route untuk menampilkan video stream
@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# Route untuk halaman utama yang menampilkan video stream di HTML
@app.route('/')
def index():
    return '''
    <html>
        <head>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {
                    margin: 0;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    background-color: #f0f0f0;
                }
                .phone-container {
                    width: 360px;
                    height: 640px;
                    background: #fff;
                    border: 16px solid #000;
                    border-radius: 36px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    position: relative;
                    overflow: hidden;
                }
                img {
                    width: 100%;
                    height: auto;
                    max-width: 360px;
                    max-height: 640px;
                }
                h1 {
                    position: absolute;
                    top: 20px;
                    width: 100%;
                    text-align: center;
                    color: #333;
                    font-size: 22px;
                    z-index: 2;
                    margin: 0;
                    padding: 0;
                }
            </style>
        </head>
        <body>
            <div class="phone-container">
                <h1>YOLO Spoof Detection Stream</h1>
                <img src="/video" alt="YOLO Spoof Detection">
            </div>
        </body>
    </html>
    '''


# Menjalankan Flask app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)