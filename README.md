# Face Anti-Spoofing Detection App

A real-time face anti-spoofing detection application built with Flask and YOLO. This application uses computer vision to detect and classify between real faces and fake faces (photos, videos, or masks) through a web interface.

## Features

- Real-time face anti-spoofing detection
- Web-based interface with mobile-friendly design
- 10-second verification process
- Visual feedback with bounding boxes and confidence scores
- Auto-reset functionality
- FPS monitoring

## Requirements

- Python 3.x
- Flask
- OpenCV (cv2)
- cvzone
- Ultralytics YOLO
- YOLO model trained for face anti-spoofing (`best_204.pt`)

## Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install flask opencv-python cvzone ultralytics
```
3. Place your trained YOLO model (`best_204.pt` or `best_190.pt`) in the `models` directory
4. Run the application:
```bash
python flask.py
```
5. Open your browser and navigate to `http://localhost:5000`

## Usage

- Position your face in front of the camera
- Wait for the 10-second verification process
- The system will identify if the face is real or fake
- Press 'R' to reset and start a new scan

## License

Copyright (c) 2024 Hamza Pratama
