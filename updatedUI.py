import math
import time
import cv2
import cvzone
from ultralytics import YOLO
from flask import Flask, Response, render_template_string

# Initialize Flask app
app = Flask(__name__)

# Initialize YOLO model
model = YOLO("../models/best_190.pt")
confidence = 0.6
classNames = ["fake", "real"]

# Open camera
cap = cv2.VideoCapture(1)  # For Webcam
cap.set(3, 360)  # Width (9 parts)
cap.set(4, 640)  # Height (16 parts)

# Calculate FPS
prev_frame_time = 0
new_frame_time = 0

# Variables for face detection tracking
face_detection_start = None
current_detection = None
verification_status = None
show_notification = False
notification_start = None
is_resetting = False
reset_countdown = 0
RESET_COUNTDOWN_DURATION = 3  # Duration of reset countdown in seconds
NOTIFICATION_DURATION = 3  # Duration of notification in seconds
VERIFICATION_THRESHOLD = 10  # Time required for verification in seconds
detect_start_time = None  # Track when detection started
verification_timer = 0  # Timer for successful detection


def advanced_detection_visualization(img, x1, y1, x2, y2, conf, detection_type):
    # Select color based on detection type
    if detection_type == 'real':
        border_color = (0, 255, 0)
        text_color = (0, 200, 0)
        bg_color = (200, 255, 200)
    else:
        border_color = (0, 0, 255)
        text_color = (0, 0, 200)
        bg_color = (255, 200, 200)

    # Gradient border effect
    for i in range(5):
        alpha = i / 5.0
        color = tuple(int(a * (1 - alpha) + b * alpha) for a, b in zip(bg_color, border_color))
        cv2.rectangle(img, (x1 - i * 2, y1 - i * 2), (x2 + i * 2, y2 + i * 2), color, max(1, 5 - i))

    # Confidence bar
    bar_width = x2 - x1
    confidence_width = int(bar_width * conf)
    cv2.rectangle(img, (x1, y2 + 10), (x2, y2 + 20), (200, 200, 200), -1)
    cv2.rectangle(img, (x1, y2 + 10), (x1 + confidence_width, y2 + 20), border_color, -1)

    # Text with shadow
    cvzone.putTextRect(
        img,
        f"{detection_type.upper()} {int(conf * 100)}%",
        (x1, y1 - 20),
        scale=1.5,
        thickness=2,
        colorR=text_color,
        colorB=(50, 50, 50),
        border=2
    )


def reset_verification():
    global face_detection_start, current_detection, verification_status, show_notification, is_resetting, reset_countdown
    global detect_start_time, verification_timer
    face_detection_start = None
    current_detection = None
    verification_status = None
    show_notification = False
    is_resetting = True
    reset_countdown = RESET_COUNTDOWN_DURATION
    detect_start_time = None
    verification_timer = 0


def start_new_verification():
    global is_resetting, reset_countdown
    is_resetting = False
    reset_countdown = 0


# Function to handle video stream
def generate_frames():
    global prev_frame_time, face_detection_start, current_detection, verification_status
    global show_notification, notification_start, is_resetting, reset_countdown
    global detect_start_time, verification_timer

    while True:
        new_frame_time = time.time()
        success, img = cap.read()

        if not success:
            break

        # Handle reset countdown
        if is_resetting:
            if reset_countdown > 0:
                cvzone.putTextRect(
                    img,
                    f"Starting new scan in: {int(reset_countdown)}s",
                    (int(img.shape[1] / 2) - 150, int(img.shape[0] / 2)),
                    scale=2, thickness=2, colorR=(255, 165, 0)
                )
                cvzone.putTextRect(
                    img,
                    "Please position your face",
                    (int(img.shape[1] / 2) - 120, int(img.shape[0] / 2) + 40),
                    scale=1, thickness=1
                )
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
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])

                if conf > confidence:
                    current_frame_detection = classNames[cls]

                    # Use advanced visualization function
                    advanced_detection_visualization(img, x1, y1, x2, y2, conf, classNames[cls])

        # Face verification logic
        if current_frame_detection:
            if current_detection != current_frame_detection:  # Detection type has changed
                current_detection = current_frame_detection
                detect_start_time = time.time()  # Start the timer for current detection
                verification_timer = 0  # Reset timer
            else:
                if detect_start_time is not None:  # Timer is running
                    elapsed_time = time.time() - detect_start_time
                    verification_timer = elapsed_time  # Update verification timer

                    # Display countdown timer for detection
                    cv2.putText(img, f"Timer: {int(verification_timer)}s",
                                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    # Draw the countdown circle
                    circle_center = (img.shape[1] - 100, 100)
                    circle_radius = 40
                    cv2.circle(img, circle_center, circle_radius, (200, 200, 200), 10)
                    angle = (verification_timer / VERIFICATION_THRESHOLD) * 360
                    cv2.ellipse(img, circle_center, (circle_radius, circle_radius), -90, 0, angle, (0, 255, 0), 10)

                    # Check if verification threshold met
                    if verification_timer >= VERIFICATION_THRESHOLD:
                        verification_status = "SUCCESS" if current_detection == "real" else "FAILED"
                        show_notification = True
                        notification_start = time.time()
        else:
            if face_detection_start and not verification_status:
                # Reset if no face detected
                cvzone.putTextRect(img, "No face detected! Please position your face.",
                                   (10, img.shape[0] - 20), scale=1, thickness=1)
                if time.time() - face_detection_start > 2:  # Reset after 2 seconds of no detection
                    reset_verification()

        # Show verification result notification
        if show_notification:
            notification_elapsed = time.time() - notification_start
            if notification_elapsed < NOTIFICATION_DURATION:
                overlay = img.copy()

                # Successful verification
                if verification_status == "SUCCESS":
                    cv2.rectangle(overlay, (0, img.shape[0] // 2 - 100),
                                  (img.shape[1], img.shape[0] // 2 + 100),
                                  (50, 200, 50), -1)
                    status_text = "VERIFICATION SUCCESSFUL"
                    status_color = (0, 255, 0)
                else:
                    # Failed verification
                    cv2.rectangle(overlay, (0, img.shape[0] // 2 - 100),
                                  (img.shape[1], img.shape[0] // 2 + 100),
                                  (50, 50, 200), -1)
                    status_text = "VERIFICATION FAILED"
                    status_color = (0, 0, 255)

                # Transparency overlay
                alpha = 0.6
                cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

                # Status text with shadow effect
                cv2.putText(img, status_text,
                            (img.shape[1] // 2 - 250, img.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0, 0, 0), 5)  # Shadow
                cv2.putText(img, status_text,
                            (img.shape[1] // 2 - 250, img.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, status_color, 3)

                # Reset button
                reset_button = (img.shape[1] // 2 - 100, img.shape[0] // 2 + 100,
                                img.shape[1] // 2 + 100, img.shape[0] // 2 + 150)
                cv2.rectangle(img, (reset_button[0], reset_button[1]),
                              (reset_button[2], reset_button[3]),
                              (150, 150, 150), -1)
                cv2.putText(img, "RESET",
                            (reset_button[0] + 50, reset_button[1] + 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 0), 2)

        # Calculate FPS
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        print(f"FPS: {fps}")

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r') or key == ord('R'):
            reset_verification()

        # Convert frame to JPEG
        ret, buffer = cv2.imencode('.jpg', img)
        if not ret:
            break
        frame = buffer.tobytes()

        yield (b'--frame\r\n'  
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Route to display video stream
@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# Route for the main page displaying video stream in HTML
@app.route('/')
def index():
    return render_template_string('''  
    <html lang="en">  
    <head>  
        <meta charset="UTF-8">  
        <meta name="viewport" content="width=device-width, initial-scale=1.0">  
        <title>Face Spoof Detection</title>  
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap" rel="stylesheet">  
        <style>  
            :root {  
                --primary-color: #3498db;  
                --secondary-color: #2ecc71;  
                --danger-color: #e74c3c;  
                --background-color: #f4f6f7;  
            }  

            * {  
                margin: 0;  
                padding: 0;  
                box-sizing: border-box;  
            }  

            body {  
                font-family: 'Roboto', sans-serif;  
                background-color: var(--background-color);  
                display: flex;  
                justify-content: center;  
                align-items: center;  
                min-height: 100vh;  
                margin: 0;  
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);  
            }  

            .container {  
                background-color: white;  
                border-radius: 20px;  
                box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);  
                overflow: hidden;  
                width: 360px;  /* Widest phone dimension */  
                height: 640px; /* Height according to 16:9 */  
                max-width: 100%;  
                position: relative;  
                border: 8px solid #000;  
                border-radius: 36px;  
            }  

            .header {  
                background-color: var(--primary-color);  
                color: white;  
                text-align: center;  
                padding: 15px;  
                position: absolute;  
                top: 0;  
                left: 0;  
                right: 0;  
                z-index: 10;  
            }  

            .header h1 {  
                font-size: 16px;  
                font-weight: 600;  
                margin: 0;  
                text-transform: uppercase;  
                letter-spacing: 1px;  
            }  

            .video-container {  
                width: 100%;  
                height: calc(100% - 60px); /* Reduce height for header */  
                position: absolute;  
                top: 60px;  
                overflow: hidden;  
            }  

            .video-stream {  
                width: 100%;  
                height: 100%;  
                object-fit: cover;  
            }  

            .status-overlay {  
                position: absolute;  
                bottom: 0;  
                left: 0;  
                right: 0;  
                background-color: rgba(0, 0, 0, 0.7);  
                color: white;  
                text-align: center;  
                padding: 10px;  
                font-weight: bold;  
            }  

            .status-success {  
                background-color: rgba(46, 204, 113, 0.8);  
            }  

            .status-danger {  
                background-color: rgba(231, 76, 60, 0.8);  
            }  

            .instructions {  
                position: absolute;  
                bottom: 0;  
                left: 0;  
                right: 0;  
                background-color: #f8f9fa;  
                border-top: 1px solid #e9ecef;  
                padding: 10px;  
                text-align: center;  
                font-size: 12px;  
                color: #6c757d;  
                z-index: 10;  
            }  

            /* Animated countdown */  
            .countdown {  
                position: absolute;  
                top: 10px;  
                right: 30px;  
                font-size: 24px;  
                color: white;  
                font-weight: bold;  
                background: rgba(0, 0, 0, 0.5);  
                padding: 5px 10px;  
                border-radius: 5px;  
                backdrop-filter: blur(5px);  
                border: 2px solid var(--primary-color);  
            }  

            /* Animation for status overlay */  
            @keyframes slideIn {  
                from {  
                    transform: translateY(100%);  
                    opacity: 0;  
                }  
                to {  
                    transform: translateY(0);  
                    opacity: 1;  
                }  
            }  

            .status-overlay.show {  
                display: block;  
                animation: slideIn 0.5s ease-out;  
            }  

            /* Responsive Design */  
            @media (max-width: 400px) {  
                .container {  
                    width: 100%;  
                    height: 100vh;  
                    border-radius: 0;  
                    border: none;  
                }  
            }  
        </style>  
    </head>  
    <body>  
        <div class="container">  
            <div class="header">  
                <h1>Liveness Detection</h1>  
            </div>  

            <div class="video-container">  
                <img src="/video" alt="Face Detection Stream" class="video-stream">  

                <div id="successOverlay" class="status-overlay status-success">  
                    Verification Successful  
                </div>  
                <div id="dangerOverlay" class="status-overlay status-danger">  
                    Verification Failed  
                </div>  
            </div>  

            <div class="instructions">  
                Keep your face steady and within the frame for verification  
            </div>  
        </div>  

        <script>  
            document.addEventListener('DOMContentLoaded', () => {  
                const successOverlay = document.getElementById('successOverlay');  
                const dangerOverlay = document.getElementById('dangerOverlay');  
                const countdownElement = document.getElementById('countdown');  

                // Function to update the countdown on the UI  
                function updateCountdown(seconds) {  
                    countdownElement.textContent = `Timer: ${seconds}s`;  
                }  

                // Function to display status  
                function showStatus(type) {  
                    // Hide all overlays  
                    successOverlay.classList.remove('show');  
                    dangerOverlay.classList.remove('show');  

                    // Show the appropriate overlay  
                    if (type === 'success') {  
                        successOverlay.classList.add('show');  
                    } else if (type === 'danger') {  
                        dangerOverlay.classList.add('show');  
                    }  

                    // Hide overlay after 3 seconds  
                    setTimeout(() => {  
                        successOverlay.classList.remove('show');  
                        dangerOverlay.classList.remove('show');  
                    }, 3000);  
                }  

                // Sample usage for debugging  
                // You can call showStatus('success') or showStatus('danger')  
                // based on the verification result from the server  
            });  
        </script>  
    </body>  
    </html>  
    ''')


# Run the Flask app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)