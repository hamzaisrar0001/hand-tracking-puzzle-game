"""
Hand Gesture Puzzle Photobooth
-------------------------------
A gesture-controlled puzzle game that uses hand tracking to:
  1. Define a capture region using both hands
  2. Capture and slice a photo into a puzzle grid
  3. Let the user solve the puzzle using pinch gestures
  4. Progress through increasing difficulty levels (3x3 -> 4x4 -> 5x5)
  5. Save the final result using a fist gesture

Controls:
  r  -> reset and restart from Level 1
  q  -> quit

Dependencies:
  pip install -r requirements.txt

Note: Sound effects use the 'winsound' module, which is Windows-only.
On other platforms, sound calls are silently skipped.
"""

import cv2
import mediapipe as mp
import numpy as np
import random
import time
import os
import threading
from datetime import datetime

try:
    import winsound
    SOUND_AVAILABLE = True
except ImportError:
    SOUND_AVAILABLE = False

# ---------------- Configuration ----------------
LEVELS = [3, 4, 5]               # grid size for each level
STABLE_MOVE_THRESHOLD = 0.015    # max allowed hand movement to count as "steady"
HOLD_SECONDS_NEEDED = 1.0        # how long hands must stay steady to lock the box
MIN_BOX_SIZE = 0.08              # minimum normalized box dimension

PINCH_THRESHOLD = 0.05           # normalized thumb-index distance for pinch detection
FIST_HOLD_SECONDS = 1.0          # how long a fist must be held to trigger save
SAVE_FOLDER = "saved_puzzles"

CELEBRATION_SECONDS = 2.5
PARTICLE_COUNT = 70

# ---------------- MediaPipe Setup ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
INDEX_MCP = 5
MIDDLE_MCP = 9
RING_MCP = 13
PINKY_MCP = 17

PARTICLE_COLORS = [
    (0, 165, 255), (0, 255, 255), (255, 0, 255),
    (0, 255, 0), (255, 255, 0), (255, 100, 100)
]


# ---------------- Sound Effects ----------------
def _beep(freq, dur):
    if SOUND_AVAILABLE:
        try:
            winsound.Beep(freq, dur)
        except Exception:
            pass


def play_async(freq, dur):
    if SOUND_AVAILABLE:
        threading.Thread(target=_beep, args=(freq, dur), daemon=True).start()


def play_sequence_async(notes):
    if not SOUND_AVAILABLE:
        return

    def run():
        for freq, dur in notes:
            _beep(freq, dur)

    threading.Thread(target=run, daemon=True).start()


def sound_pick():
    play_async(1000, 50)


def sound_drop():
    play_async(650, 70)


def sound_level_complete():
    play_sequence_async([(600, 90), (800, 90), (1000, 90), (1300, 140)])


def sound_game_complete():
    play_sequence_async([(700, 90), (900, 90), (1100, 90), (1300, 90), (1600, 220)])


# ---------------- Application State ----------------
class AppState:
    def __init__(self):
        self.locked_box = None
        self.prev_corners = None
        self.stable_since = None

        self.level_index = 0
        self.original_crop = None
        self.grid = LEVELS[0]

        self.pieces = None
        self.piece_size = None

        self.held_slot = None
        self.drag_pos = None
        self.was_pinching = False

        self.solved = False
        self.fist_since = None
        self.saved_this_solve = False

        self.celebrating = False
        self.celebration_start = None
        self.particles = []
        self.game_complete = False

        self.last_time = time.time()

    def reset(self):
        self.__init__()


state = AppState()


# ---------------- Helper Functions ----------------
def dist(a, b):
    return np.hypot(a[0] - b[0], a[1] - b[1])


def normalize_box(a, b):
    x = min(a[0], b[0]); y = min(a[1], b[1])
    w = abs(a[0] - b[0]); h = abs(a[1] - b[1])
    return x, y, w, h


def norm_box_to_px(box, frame_w, frame_h):
    x, y, w, h = box
    return (int(x * frame_w), int(y * frame_h), int(w * frame_w), int(h * frame_h))


def capture_crop(frame, box_px):
    x, y, w, h = box_px
    x = max(0, x); y = max(0, y)
    w = max(LEVELS[-1], w); h = max(LEVELS[-1], h)
    return frame[y:y + h, x:x + w].copy()


def slice_pieces(crop, grid):
    h, w = crop.shape[:2]
    pw = w // grid
    ph = h // grid
    resized = cv2.resize(crop, (pw * grid, ph * grid))

    pieces = []
    for row in range(grid):
        for col in range(grid):
            piece = resized[row * ph:(row + 1) * ph, col * pw:(col + 1) * pw].copy()
            correct_index = row * grid + col
            pieces.append((piece, correct_index))

    shuffled = pieces[:]
    while grid > 1 and check_solved(shuffled):
        random.shuffle(shuffled)
    random.shuffle(shuffled)
    return shuffled, (pw, ph)


def start_level(grid):
    pieces, piece_size = slice_pieces(state.original_crop, grid)
    state.grid = grid
    state.pieces = pieces
    state.piece_size = piece_size
    state.held_slot = None
    state.drag_pos = None
    state.was_pinching = False
    state.solved = False
    state.saved_this_solve = False


def is_pinching(landmarks):
    thumb = landmarks.landmark[THUMB_TIP]
    index = landmarks.landmark[INDEX_TIP]
    d = dist((thumb.x, thumb.y), (index.x, index.y))
    pinch_point = ((thumb.x + index.x) / 2, (thumb.y + index.y) / 2)
    return d < PINCH_THRESHOLD, pinch_point


def is_fist(landmarks):
    tips = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    mcps = [INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
    closed = 0
    for tip, mcp in zip(tips, mcps):
        t = landmarks.landmark[tip]
        m = landmarks.landmark[mcp]
        if dist((t.x, t.y), (m.x, m.y)) < 0.08:
            closed += 1
    return closed >= 3


def slot_at_point(point_px, box_px, piece_size, grid):
    x, y, w, h = box_px
    pw, ph = piece_size
    px, py = point_px
    if not (x <= px <= x + w and y <= py <= y + h):
        return None
    col = int((px - x) // pw)
    row = int((py - y) // ph)
    col = min(max(col, 0), grid - 1)
    row = min(max(row, 0), grid - 1)
    return row * grid + col


def check_solved(pieces):
    return all(pieces[i][1] == i for i in range(len(pieces)))


def spawn_confetti(box_px):
    x, y, w, h = box_px
    cx, cy = x + w // 2, y + h // 2
    particles = []
    for _ in range(PARTICLE_COUNT):
        angle = random.uniform(0, 2 * np.pi)
        speed = random.uniform(2, 7)
        particles.append({
            "x": float(cx), "y": float(cy),
            "vx": np.cos(angle) * speed,
            "vy": np.sin(angle) * speed - 3,
            "color": random.choice(PARTICLE_COLORS),
            "radius": random.randint(3, 6),
            "life": 1.0,
        })
    return particles


def update_and_draw_particles(frame, particles, dt):
    for p in particles:
        p["x"] += p["vx"]
        p["y"] += p["vy"]
        p["vy"] += 0.15
        p["life"] -= dt / CELEBRATION_SECONDS
        if p["life"] > 0:
            r = max(1, int(p["radius"] * p["life"]))
            cv2.circle(frame, (int(p["x"]), int(p["y"])), r, p["color"], -1)


def draw_puzzle(frame, box_px, pieces, piece_size, grid, held_slot, drag_pos, solved,
                 hover_slot=None, cursor_px=None, cursor_active=False):
    x, y, w, h = box_px
    pw, ph = piece_size

    for slot, (piece_img, correct_index) in enumerate(pieces):
        if slot == held_slot:
            continue
        row = slot // grid
        col = slot % grid
        px = x + col * pw
        py = y + row * ph
        ph_img, pw_img = piece_img.shape[:2]
        if py + ph_img > frame.shape[0] or px + pw_img > frame.shape[1]:
            continue
        frame[py:py + ph_img, px:px + pw_img] = piece_img
        border_color = (0, 255, 0) if solved else (0, 200, 255)
        cv2.rectangle(frame, (px, py), (px + pw_img, py + ph_img), border_color, 1)

    # Highlight the tile currently under the cursor / dragged piece
    if hover_slot is not None and hover_slot != held_slot:
        row = hover_slot // grid
        col = hover_slot % grid
        hx = x + col * pw
        hy = y + row * ph
        overlay = frame.copy()
        highlight_color = (255, 0, 255) if held_slot is not None else (0, 255, 255)
        cv2.rectangle(overlay, (hx, hy), (hx + pw, hy + ph), highlight_color, -1)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        cv2.rectangle(frame, (hx, hy), (hx + pw, hy + ph), highlight_color, 3)

    # Draw the dragged piece last so it renders on top
    if held_slot is not None and drag_pos is not None:
        piece_img = pieces[held_slot][0]
        ph_img, pw_img = piece_img.shape[:2]
        dx = int(drag_pos[0] - pw_img / 2)
        dy = int(drag_pos[1] - ph_img / 2)
        dx = max(0, min(dx, frame.shape[1] - pw_img))
        dy = max(0, min(dy, frame.shape[0] - ph_img))
        frame[dy:dy + ph_img, dx:dx + pw_img] = piece_img
        cv2.rectangle(frame, (dx, dy), (dx + pw_img, dy + ph_img), (0, 165, 255), 3)

    box_color = (0, 255, 0) if solved else (255, 255, 0)
    cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 3)

    if cursor_px is not None:
        cursor_color = (0, 140, 255) if cursor_active else (255, 255, 255)
        cv2.circle(frame, cursor_px, 14, cursor_color, 2)
        cv2.circle(frame, cursor_px, 3, cursor_color, -1)


def draw_centered_text(frame, text, y, scale, color, thickness):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = (frame.shape[1] - tw) // 2
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def main():
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: could not open the camera.")
        return

    print("Controls: 'r' = reset, 'q' = quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.time()
        dt = now - state.last_time
        state.last_time = now

        frame = cv2.flip(frame, 1)
        frame_h, frame_w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        hands_lm = result.multi_hand_landmarks or []

        # ================= Celebration State =================
        if state.celebrating:
            draw_puzzle(frame, state.locked_box, state.pieces, state.piece_size,
                        state.grid, None, None, True)
            update_and_draw_particles(frame, state.particles, dt)

            if state.game_complete:
                draw_centered_text(frame, "ALL LEVELS COMPLETE!", 60, 1.0, (0, 255, 0), 3)
                draw_centered_text(frame, "Make a fist to save", 95, 0.65, (0, 255, 255), 2)
            else:
                draw_centered_text(frame, f"LEVEL {state.level_index + 1} COMPLETE!",
                                    60, 0.9, (0, 255, 0), 3)
                next_grid = LEVELS[state.level_index + 1]
                draw_centered_text(frame, f"Next: {next_grid}x{next_grid}", 95, 0.65,
                                    (0, 255, 255), 2)

            if state.game_complete:
                fist_active = any(is_fist(lm) for lm in hands_lm)
                if fist_active:
                    if state.fist_since is None:
                        state.fist_since = now
                    if now - state.fist_since >= FIST_HOLD_SECONDS and not state.saved_this_solve:
                        filename = os.path.join(
                            SAVE_FOLDER,
                            f"puzzle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                        )
                        cv2.imwrite(filename, frame)
                        state.saved_this_solve = True
                        print(f"Saved: {filename}")
                else:
                    state.fist_since = None
                if state.saved_this_solve:
                    draw_centered_text(frame, "Saved!", 130, 0.7, (0, 255, 255), 2)

            if now - state.celebration_start >= CELEBRATION_SECONDS:
                state.celebrating = False
                if not state.game_complete:
                    state.level_index += 1
                    if state.level_index >= len(LEVELS):
                        state.game_complete = True
                        state.celebrating = True
                        state.celebration_start = time.time()
                        state.particles = spawn_confetti(state.locked_box)
                        sound_game_complete()
                    else:
                        start_level(LEVELS[state.level_index])

        # ================= Puzzle Solving State =================
        elif state.locked_box is not None:
            pinch_active = False
            pinch_px = None
            fist_active = False
            cursor_px = None

            for lm in hands_lm:
                tip = lm.landmark[INDEX_TIP]
                cursor_px = (int(tip.x * frame_w), int(tip.y * frame_h))
                pinching, pinch_point = is_pinching(lm)
                if pinching:
                    pinch_active = True
                    pinch_px = (int(pinch_point[0] * frame_w), int(pinch_point[1] * frame_h))
                if is_fist(lm):
                    fist_active = True

            live_cursor = pinch_px if pinch_active else cursor_px
            hover_slot = slot_at_point(live_cursor, state.locked_box, state.piece_size, state.grid) \
                if live_cursor else None

            if pinch_active and not state.was_pinching and state.held_slot is None:
                slot = slot_at_point(pinch_px, state.locked_box, state.piece_size, state.grid)
                if slot is not None:
                    state.held_slot = slot
                    sound_pick()

            if pinch_active and state.held_slot is not None:
                state.drag_pos = pinch_px

            if not pinch_active and state.was_pinching and state.held_slot is not None:
                drop_slot = slot_at_point(state.drag_pos, state.locked_box, state.piece_size, state.grid) \
                    if state.drag_pos else None
                if drop_slot is not None and drop_slot != state.held_slot:
                    pieces = state.pieces
                    pieces[state.held_slot], pieces[drop_slot] = pieces[drop_slot], pieces[state.held_slot]
                    sound_drop()
                    just_solved = check_solved(pieces)
                    if just_solved and not state.solved:
                        state.solved = True
                        state.celebrating = True
                        state.celebration_start = time.time()
                        state.particles = spawn_confetti(state.locked_box)
                        sound_level_complete()
                state.held_slot = None
                state.drag_pos = None

            state.was_pinching = pinch_active

            draw_puzzle(frame, state.locked_box, state.pieces, state.piece_size, state.grid,
                        state.held_slot, state.drag_pos, state.solved,
                        hover_slot=hover_slot, cursor_px=live_cursor, cursor_active=pinch_active)

            level_label = f"Level {state.level_index + 1}/{len(LEVELS)}  ({state.grid}x{state.grid})"
            cv2.putText(frame, level_label, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, "Pinch a piece and drop it on another slot to swap", (10, frame_h - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, "'r' = reset", (10, frame_h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # ================= Box Drawing State =================
        elif len(hands_lm) >= 2:
            cornerA = hands_lm[0].landmark[INDEX_TIP]
            cornerB = hands_lm[1].landmark[INDEX_TIP]
            a = (cornerA.x, cornerA.y)
            b = (cornerB.x, cornerB.y)
            box = normalize_box(a, b)

            ax, ay = int(a[0] * frame_w), int(a[1] * frame_h)
            bx, by = int(b[0] * frame_w), int(b[1] * frame_h)
            cv2.circle(frame, (ax, ay), 10, (0, 255, 255), -1)
            cv2.circle(frame, (bx, by), 10, (0, 255, 255), -1)

            box_px = norm_box_to_px(box, frame_w, frame_h)
            cv2.rectangle(frame, (box_px[0], box_px[1]),
                          (box_px[0] + box_px[2], box_px[1] + box_px[3]), (0, 165, 255), 2)

            if state.prev_corners is not None:
                moveA = dist(a, state.prev_corners[0])
                moveB = dist(b, state.prev_corners[1])
                is_stable = moveA < STABLE_MOVE_THRESHOLD and moveB < STABLE_MOVE_THRESHOLD
                if is_stable:
                    if state.stable_since is None:
                        state.stable_since = now
                else:
                    state.stable_since = None
            state.prev_corners = (a, b)

            if state.stable_since is not None:
                held = now - state.stable_since
                progress = min(held / HOLD_SECONDS_NEEDED, 1.0)
                bar_w = int(200 * progress)
                cv2.rectangle(frame, (10, 50), (10 + 200, 65), (60, 60, 60), -1)
                cv2.rectangle(frame, (10, 50), (10 + bar_w, 65), (0, 140, 255), -1)
                cv2.putText(frame, f"Hold steady: {int(progress * 100)}%", (10, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

                if progress >= 1.0 and box[2] > MIN_BOX_SIZE and box[3] > MIN_BOX_SIZE:
                    state.locked_box = box_px
                    state.original_crop = capture_crop(frame, box_px)
                    state.level_index = 0
                    start_level(LEVELS[0])
            else:
                cv2.putText(frame, "Hold both hands steady to lock the box", (10, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # ================= Waiting for Hands =================
        else:
            state.prev_corners = None
            state.stable_since = None
            msg = "One hand detected, show the other" if len(hands_lm) == 1 else "Show both hands to begin"
            cv2.putText(frame, msg, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            for lm in hands_lm:
                p = lm.landmark[INDEX_TIP]
                cv2.circle(frame, (int(p.x * frame_w), int(p.y * frame_h)), 8, (0, 255, 0), -1)

        cv2.imshow("Hand Gesture Puzzle Photobooth", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            state.reset()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
