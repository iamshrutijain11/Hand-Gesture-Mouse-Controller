# Hand Gesture Control System

A computer vision-based hand gesture control system that allows you to control media playback and navigate YouTube using hand gestures captured through your webcam.

## Features

### Media Control (`Openwebcam.py`)
- **Previous Track**: Move index finger to the left edge of the screen
- **Next Track**: Move index finger to the right edge of the screen
- Visual feedback with control zones and gesture detection
- Debounce protection to prevent rapid-fire actions

### YouTube Control (`youtubecontrol.py`)
- **Cursor Movement**: Point with index finger to control mouse cursor
- **Click/Play/Pause**: Three-finger pinch gesture (index, middle, ring fingers)
- **Scroll Up**: Two fingers up and close together
- **Scroll Down**: Two fingers down and close together
- **Skip Backward**: Index finger in left zone (10 seconds back)
- **Skip Forward**: Index finger in right zone (10 seconds forward)

## Requirements

```
opencv-python
mediapipe
keyboard
pyautogui
numpy
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/[username]/hand-gesture-control.git
cd hand-gesture-control
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Media Control
```bash
python Openwebcam.py
```

### YouTube Control
```bash
python youtubecontrol.py
```

Press 'q' to exit either application.

## How It Works

The system uses:
- **OpenCV** for webcam capture and image processing
- **MediaPipe** for hand landmark detection and tracking
- **PyAutoGUI** for system control (mouse, keyboard, scrolling)
- **Keyboard** library for media key simulation

## Gesture Recognition

The applications detect hand landmarks in real-time and interpret finger positions to trigger different actions. Control zones are defined as percentages of the frame width/height for consistent gesture recognition across different webcam resolutions.

## Notes

- Ensure good lighting for optimal hand detection
- Keep your hand clearly visible in the webcam frame
- The system includes debounce delays to prevent accidental repeated actions
- Works best with a stable webcam setup

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the [MIT License](LICENSE).