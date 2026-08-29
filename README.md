# 🧩 Hand Gesture Puzzle Photobooth

An AI-powered, gesture-controlled puzzle photobooth built with **Python**, **OpenCV**, and **MediaPipe**. Users define a capture region with two-hand tracking, get their photo sliced into a puzzle, and solve it entirely through hand gestures — no mouse, no touchscreen, no keyboard required for gameplay.

## ✨ Features

- **Two-hand box detection** — define a custom capture region by pointing with both index fingers; the box locks automatically once your hands hold steady
- **Automatic photo capture & slicing** — the captured frame is sliced into a puzzle grid
- **Gesture-based puzzle solving** — pinch to pick up a piece, drag it with your hand, and release over another tile to swap
- **Progressive difficulty levels** — solve 3×3, then automatically advance to 4×4, then 5×5
- **Celebration effects** — confetti animation and on-screen text when each level is completed
- **Sound feedback** — distinct audio cues for picking up a piece, dropping/swapping, and completing a level
- **Gesture-based save** — make a fist after completing all levels to save the final result as an image
- **Live hover cursor** — a visual indicator always shows which tile your hand is currently over

## 🛠️ Tech Stack

| Component | Purpose |
|---|---|
| [Python 3.10](https://www.python.org/) | Core programming language |
| [OpenCV](https://opencv.org/) | Webcam capture, image processing, and rendering |
| [MediaPipe](https://developers.google.com/mediapipe) | Real-time hand landmark detection (pre-trained ML model) |
| [NumPy](https://numpy.org/) | Coordinate math and array operations |
| `winsound` (Python standard library) | Sound effects on Windows |

## 📋 Requirements

- Python 3.9–3.11 (tested on 3.10)
- A webcam
- Windows (for sound effects — the app still runs on other platforms, but audio cues will be silently skipped)

## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```

2. **Create and activate a virtual environment (recommended)**
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   python puzzle_photobooth_final.py
   ```

## 🎮 How to Play

1. Hold up **both hands** in front of the camera with your index fingers pointing — this sets the two corners of your capture box.
2. Keep your hands **steady for about a second**; the box will lock automatically and your photo will be captured and sliced.
3. **Pinch** (touch your thumb and index finger together) over a puzzle piece to pick it up.
4. Move your hand while pinching to **drag** the piece, then release the pinch over another tile to **swap** the two pieces.
5. Solve the puzzle to trigger a celebration and automatically advance to the next difficulty level.
6. After completing all levels, **make a fist** and hold it for about a second to save your result.

### Keyboard Controls

| Key | Action |
|---|---|
| `r` | Reset and start over from Level 1 |
| `q` | Quit the application |

## 📁 Project Structure

```
├── puzzle_photobooth_final.py   # Main application
├── requirements.txt              # Python dependencies
└── saved_puzzles/                # Auto-created folder where completed puzzles are saved
```

## 📸 Output

Completed puzzles are saved automatically to the `saved_puzzles/` folder as timestamped PNG files (e.g. `puzzle_20260828_143022.png`).


## 📄 License

This project is open source and available for personal and educational use.
