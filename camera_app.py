#!/usr/bin/env python3
"""USB Camera Web Interface for Raspberry Pi with human detection."""

import threading
import time
import os
import io
from flask import Flask, Response, render_template_string, jsonify, request, send_file
import cv2
import numpy as np
from gpiozero import OutputDevice
from adafruit_servokit import ServoKit

app = Flask(__name__)

# --- Flywheel relay on BCM GPIO 17 (BOARD pin 11) ---
# Relay is active-low: pin LOW = relay ON, pin HIGH = relay OFF
_flywheel_relay = OutputDevice(17, active_high=False, initial_value=False)
flywheel_on = False

# --- Trigger relay on BCM GPIO 27 (BOARD pin 13) ---
# Relay is active-low: pin LOW = relay ON, pin HIGH = relay OFF
_trigger_relay = OutputDevice(27, active_high=False, initial_value=False)
trigger_on = False

# --- PCA9685 Servos on channels 0 and 2 ---
try:
    _servo_kit = ServoKit(channels=16)
    _servo_kit.servo[0].angle = 0
    _servo_kit.servo[2].angle = 0
except (ValueError, OSError) as e:
    print(f"WARNING: PCA9685 servo controller not found ({e}). Servo disabled.")
    _servo_kit = None
servo_angles = {0: 0, 2: 0}
_servo_lock = threading.Lock()

# --- Configuration ---
CONFIDENCE_THRESHOLD = 0.5
PERSON_CLASS = 15  # "person" in MobileNet SSD VOC classes
DETECTION_HOLD_FRAMES = 1

# --- Load model ---
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
net = cv2.dnn.readNetFromCaffe(
    os.path.join(MODEL_DIR, 'MobileNetSSD_deploy.prototxt'),
    os.path.join(MODEL_DIR, 'MobileNetSSD_deploy.caffemodel')
)

# --- Shared state ---
zoom_level = 1.0
latest_frame = None
frame_lock = threading.Lock()
detection_boxes = []
detection_lock = threading.Lock()
human_detected = False
human_count = 0
target_centered = False
aim_direction = ""  # e.g. "left", "up right", etc.


def open_camera():
    """Try to open a working USB camera, return the VideoCapture or None."""
    for idx in range(5):
        c = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if c.isOpened():
            c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            c.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            c.set(cv2.CAP_PROP_FPS, 30)
            c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, _ = c.read()
            if ret:
                return c
            c.release()
    return None


def camera_reader():
    """Dedicated thread: reads frames as fast as possible, keeps only the latest.
    Auto-recovers if the camera stops delivering frames."""
    global latest_frame
    MAX_FAILURES = 30  # ~10s of consecutive failures before reconnect

    while True:
        cap = open_camera()
        if cap is None:
            print("WARNING: No working camera found, retrying in 3s...")
            time.sleep(3)
            continue

        print("Camera opened successfully.")
        fail_count = 0
        while True:
            ret, frame = cap.read()
            if ret:
                fail_count = 0
                frame = cv2.flip(frame, -1)  # flip 180 degrees (upside down camera)
                with frame_lock:
                    latest_frame = frame
            else:
                fail_count += 1
                if fail_count >= MAX_FAILURES:
                    print(f"Camera failed {MAX_FAILURES} consecutive reads, reconnecting...")
                    break
                time.sleep(0.03)

        cap.release()
        time.sleep(1)  # brief pause before reconnect


def detection_worker():
    """Dedicated thread: runs person detection on latest frame without blocking streaming."""
    global detection_boxes, human_detected, human_count, target_centered, aim_direction
    hold = 0

    while True:
        with frame_lock:
            frame = latest_frame

        if frame is None:
            time.sleep(0.05)
            continue

        # Apply current zoom before detection
        zoomed = apply_zoom(frame, zoom_level)
        h, w = zoomed.shape[:2]
        blob = cv2.dnn.blobFromImage(zoomed, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            class_id = int(detections[0, 0, i, 1])
            if class_id == PERSON_CLASS and confidence > CONFIDENCE_THRESHOLD:
                x1 = max(0, int(detections[0, 0, i, 3] * w))
                y1 = max(0, int(detections[0, 0, i, 4] * h))
                x2 = min(w, int(detections[0, 0, i, 5] * w))
                y2 = min(h, int(detections[0, 0, i, 6] * h))
                boxes.append((x1, y1, x2 - x1, y2 - y1))

        # Check if any target head is near center of frame (~20% threshold)
        centered = False
        direction = ""
        cx, cy = w // 2, h // 2
        thresh_x, thresh_y = w // 5, h // 5
        if len(boxes) > 0:
            # Use the first (most confident) detection
            bx, by, bw, bh = boxes[0]
            head_x = bx + bw // 2
            head_y = by + int(bh * 0.1)
            dx = head_x - cx
            dy = head_y - cy
            if abs(dx) < thresh_x and abs(dy) < thresh_y:
                centered = True
            else:
                parts = []
                if dy < -thresh_y:
                    parts.append("up")
                elif dy > thresh_y:
                    parts.append("down")
                if dx < -thresh_x:
                    parts.append("left")
                elif dx > thresh_x:
                    parts.append("right")
                direction = " ".join(parts)

        with detection_lock:
            if len(boxes) > 0:
                detection_boxes = boxes
                human_detected = True
                human_count = len(boxes)
                hold = DETECTION_HOLD_FRAMES
                target_centered = centered
                aim_direction = direction
            else:
                target_centered = False
                aim_direction = ""
                if hold > 0:
                    hold -= 1
                else:
                    detection_boxes = []
                    human_detected = False
                    human_count = 0

        time.sleep(0.03)  # run as fast as inference allows


def apply_zoom(frame, level):
    """Crop center of frame based on zoom level, then resize back."""
    if level <= 1.0:
        return frame
    h, w = frame.shape[:2]
    crop_w = int(w / level)
    crop_h = int(h / level)
    x1 = (w - crop_w) // 2
    y1 = (h - crop_h) // 2
    cropped = frame[y1:y1 + crop_h, x1:x1 + crop_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def servo_sweep_worker():
    """No-op — servo is now controlled directly by the endpoint."""
    while True:
        time.sleep(1)


def draw_target(frame, head_x, head_y, r):
    """Draw a target reticle at the given position."""
    cv2.circle(frame, (head_x, head_y), r, (0, 0, 255), 2)
    cv2.circle(frame, (head_x, head_y), r * 2 // 3, (0, 0, 255), 1)
    cv2.circle(frame, (head_x, head_y), r // 3, (0, 0, 255), 2)
    gap = r + 6
    ext = r + 16
    cv2.line(frame, (head_x - ext, head_y), (head_x - gap, head_y), (0, 0, 255), 2)
    cv2.line(frame, (head_x + gap, head_y), (head_x + ext, head_y), (0, 0, 255), 2)
    cv2.line(frame, (head_x, head_y - ext), (head_x, head_y - gap), (0, 0, 255), 2)
    cv2.line(frame, (head_x, head_y + gap), (head_x, head_y + ext), (0, 0, 255), 2)


def generate_frames():
    """Generator that yields MJPEG frames with overlays. No detection here — just drawing."""
    while True:
        with frame_lock:
            frame = latest_frame

        if frame is None:
            time.sleep(0.03)
            continue

        frame = apply_zoom(frame.copy(), zoom_level)

        # Draw detection overlays from the detection thread
        with detection_lock:
            boxes = detection_boxes.copy()
            detected = human_detected
            count = human_count

        if detected and boxes:
            for (x, y, w, h) in boxes:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                head_x = x + w // 2
                head_y = y + int(h * 0.1)
                r = max(18, w // 6)
                draw_target(frame, head_x, head_y, r)

            label = f"HUMAN DETECTED ({count})"
            cv2.putText(frame, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # Timestamp
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, ts, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Zoom indicator
        if zoom_level > 1.0:
            cv2.putText(frame, f"{zoom_level:.1f}x", (frame.shape[1] - 80, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.033)  # ~30fps cap to prevent flooding


@app.route('/')
def index():
    return render_template_string(HTML_PAGE)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/frame')
def single_frame():
    """Return a single JPEG frame with overlays."""
    with frame_lock:
        frame = latest_frame
    if frame is None:
        return "No frame", 503
    frame = apply_zoom(frame.copy(), zoom_level)

    with detection_lock:
        boxes = detection_boxes.copy()
        detected = human_detected
        count = human_count

    if detected and boxes:
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            head_x = x + w // 2
            head_y = y + int(h * 0.1)
            r = max(18, w // 6)
            draw_target(frame, head_x, head_y, r)
        label = f"HUMAN DETECTED ({count})"
        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, ts, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if zoom_level > 1.0:
        cv2.putText(frame, f"{zoom_level:.1f}x", (frame.shape[1] - 80, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return Response(buf.tobytes(), mimetype='image/jpeg',
                    headers={
                        'Cache-Control': 'no-store',
                        'X-Human': '1' if detected else '0',
                        'X-Count': str(count),
                        'X-Centered': '1' if (detected and target_centered) else '0',
                        'X-Direction': aim_direction if (detected and not target_centered) else '',
                        'Access-Control-Expose-Headers': 'X-Human, X-Count, X-Centered, X-Direction',
                    })


@app.route('/sounds/<filename>')
def serve_sound(filename):
    """Serve voice clip WAV files."""
    sounds_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sounds')
    return send_file(os.path.join(sounds_dir, filename), mimetype='audio/wav')


@app.route('/set_zoom', methods=['POST'])
def set_zoom():
    global zoom_level
    data = request.get_json(force=True)
    zoom_level = max(1.0, min(5.0, float(data.get('zoom', 1.0))))
    return jsonify(zoom=zoom_level)


@app.route('/snapshot')
def snapshot():
    with frame_lock:
        frame = latest_frame
    if frame is None:
        return "Camera error", 500
    frame = apply_zoom(frame.copy(), zoom_level)
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return send_file(
        io.BytesIO(buf.tobytes()),
        mimetype='image/jpeg',
        as_attachment=True,
        download_name=f'snapshot_{time.strftime("%Y%m%d_%H%M%S")}.jpg'
    )


@app.route('/flywheel', methods=['POST'])
def flywheel_control():
    global flywheel_on
    data = request.get_json(force=True)
    state = bool(data.get('state', False))
    if state:
        _flywheel_relay.on()
    else:
        _flywheel_relay.off()
    flywheel_on = state
    return jsonify(flywheel=flywheel_on)


@app.route('/trigger', methods=['POST'])
def trigger_control():
    global trigger_on
    data = request.get_json(force=True)
    state = bool(data.get('state', False))
    if state:
        _trigger_relay.on()
    else:
        _trigger_relay.off()
    trigger_on = state
    return jsonify(trigger=trigger_on)


@app.route('/servo', methods=['POST'])
def servo_control():
    data = request.get_json(force=True)
    channel = int(data.get('channel', 0))
    if channel not in (0, 2):
        return jsonify(error='Invalid channel'), 400
    if _servo_kit is None:
        return jsonify(angle=0, error='Servo controller not connected')
    action = data.get('action', 'set')
    angle = max(0, min(180, float(data.get('angle', 0))))
    with _servo_lock:
        _servo_kit.servo[channel].angle = angle
    servo_angles[channel] = angle
    return jsonify(channel=channel, angle=angle)


@app.route('/status')
def status():
    return jsonify(
        zoom=zoom_level,
        human_detected=human_detected,
        human_count=human_count,
        target_centered=target_centered,
        flywheel=flywheel_on,
        trigger=trigger_on,
        servo_angles=servo_angles
    )


# ─── Inline HTML/CSS/JS ───

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Camera Control</title>
<style>
  @keyframes rainbow-bg {
    0%   { background-position: 0% 50%; }
    100% { background-position: 200% 50%; }
  }
  @keyframes rainbow-text {
    0%   { color: #ff0000; }
    16%  { color: #ff8800; }
    33%  { color: #ffff00; }
    50%  { color: #00cc00; }
    66%  { color: #0088ff; }
    83%  { color: #8800ff; }
    100% { color: #ff0000; }
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: linear-gradient(135deg, #1a0a2e, #0a1a2e, #0a2e1a, #2e1a0a, #2e0a1a);
    background-size: 400% 400%;
    animation: rainbow-bg 12s ease infinite;
    color: #eee;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    display: flex; flex-direction: column; height: 100vh;
  }
  header {
    background: linear-gradient(90deg, #ff0000, #ff8800, #ffff00, #00cc00, #0088ff, #8800ff, #ff0000);
    background-size: 200% 100%;
    animation: rainbow-bg 4s linear infinite;
    padding: 8px 16px;
    display: flex; align-items: center; justify-content: space-between;
    border-bottom: 2px solid rgba(255,255,255,0.3);
  }
  header h1 { font-size: 1.1rem; font-weight: 600; color: #fff; text-shadow: 0 0 8px rgba(0,0,0,0.5); }
  #status-badge {
    padding: 4px 10px; border-radius: 12px; font-size: 0.75rem;
    background: rgba(0,0,0,0.4); color: #8f8;
  }
  #status-badge.alert { background: #e94560; color: #fff; }
  .feed-container {
    flex: 1; display: flex; align-items: center; justify-content: center;
    overflow: hidden; background: #000;
    border: 2px solid;
    border-image: linear-gradient(90deg, #ff0000, #ff8800, #ffff00, #00cc00, #0088ff, #8800ff) 1;
  }
  .feed-container img {
    max-width: 100%; max-height: 100%; object-fit: contain;
  }
  .controls {
    background: linear-gradient(90deg, rgba(255,0,0,0.15), rgba(255,136,0,0.15), rgba(255,255,0,0.15), rgba(0,204,0,0.15), rgba(0,136,255,0.15), rgba(136,0,255,0.15));
    padding: 12px 16px;
    display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
    border-top: 2px solid;
    border-image: linear-gradient(90deg, #ff0000, #ff8800, #ffff00, #00cc00, #0088ff, #8800ff) 1;
  }
  .controls label { font-size: 0.85rem; animation: rainbow-text 3s linear infinite; }
  .controls input[type=range] {
    width: 160px; accent-color: #ff8800;
  }
  .zoom-val {
    font-size: 0.85rem; min-width: 36px;
    animation: rainbow-text 2s linear infinite;
  }
  .btn {
    padding: 8px 18px; border: none; border-radius: 6px;
    cursor: pointer; font-size: 0.85rem; font-weight: 500;
    transition: opacity 0.2s, transform 0.1s;
  }
  .btn:hover { opacity: 0.85; transform: scale(1.05); }
  .btn-snap {
    background: linear-gradient(135deg, #0088ff, #8800ff); color: #fff;
  }
  .btn-sound { background: linear-gradient(135deg, #ff8800, #ff0000); color: #fff; }
  .btn-sound.enabled { background: linear-gradient(135deg, #00cc00, #008800); }
  .btn-flywheel {
    background: linear-gradient(135deg, #0088ff, #00cc00); color: #fff;
    font-size: 1rem; padding: 10px 22px;
  }
  .btn-flywheel.on { background: linear-gradient(135deg, #00ff00, #00cc00); box-shadow: 0 0 16px rgba(0,255,0,0.5); }
  .btn-trigger {
    background: linear-gradient(135deg, #ff0000, #ff8800); color: #fff;
    font-size: 1rem; padding: 10px 22px;
  }
  .btn-trigger.on { background: linear-gradient(135deg, #ff0000, #ff0044); box-shadow: 0 0 16px rgba(255,0,0,0.6); }
  .btn-zoom {
    background: linear-gradient(135deg, #8800ff, #0088ff); color: #fff;
    width: 36px; text-align: center;
  }
  .zoom-row {
    display: flex; align-items: center; gap: 8px; width: 100%;
    flex-wrap: nowrap;
  }
  .zoom-row input[type=range] { flex: 1; min-width: 80px; }
  .joystick-wrap {
    display: flex; align-items: center; gap: 16px;
  }
  .joystick-container {
    position: relative; width: 150px; height: 150px;
    background: conic-gradient(#ff0000, #ff8800, #ffff00, #00cc00, #0088ff, #8800ff, #ff0000);
    border-radius: 50%; border: 3px solid rgba(255,255,255,0.3);
    touch-action: none; user-select: none;
  }
  .joystick-knob {
    position: absolute; width: 40px; height: 40px;
    background: radial-gradient(circle, #fff, #ddd);
    border: 2px solid rgba(0,0,0,0.3);
    border-radius: 50%;
    top: 50%; left: 50%; transform: translate(-50%, -50%);
    pointer-events: none;
    box-shadow: 0 0 10px rgba(0,0,0,0.4);
  }
  .joystick-label { font-size: 0.75rem; animation: rainbow-text 3s linear infinite; text-align: center; }
  @media (max-width: 600px) {
    .controls { justify-content: center; }
    .controls input[type=range] { width: 120px; }
    .joystick-container { width: 120px; height: 120px; }
    .joystick-knob { width: 32px; height: 32px; }
  }
</style>
</head>
<body>
  <header>
    <h1>Camera Control</h1>
    <span id="status-badge">Monitoring</span>
  </header>

  <div class="feed-container">
    <img id="feed" alt="Live Feed">
  </div>

  <div class="controls">
    <button class="btn btn-flywheel" id="flywheel-btn" onclick="toggleFlywheel()">FLYWHEEL</button>
    <button class="btn btn-trigger" id="trigger-btn"
            onmousedown="triggerStart()" onmouseup="triggerStop()" onmouseleave="triggerStop()"
            ontouchstart="triggerStart(event)" ontouchend="triggerStop()" ontouchcancel="triggerStop()">TRIGGER</button>

    <div class="joystick-wrap">
      <div>
        <div class="joystick-container" id="joystick">
          <div class="joystick-knob" id="joystick-knob"></div>
        </div>
        <div class="joystick-label">L/R: pan &middot; U/D: tilt</div>
      </div>
    </div>

    <div class="zoom-row">
      <button class="btn btn-zoom" onclick="changeZoom(-0.5)">&#x2212;</button>
      <label>Zoom</label>
      <input type="range" id="zoom-slider" min="1" max="5" step="0.1" value="1"
             oninput="setZoom(this.value)">
      <span class="zoom-val" id="zoom-val">1.0x</span>
      <button class="btn btn-zoom" onclick="changeZoom(0.5)">+</button>
    </div>
    <button class="btn btn-snap" onclick="takeSnapshot()">Snapshot</button>
    <button class="btn btn-sound" id="sound-btn" onclick="enableSound()">Enable Sound</button>
  </div>

<script>
  const slider = document.getElementById('zoom-slider');
  const zoomVal = document.getElementById('zoom-val');
  const badge = document.getElementById('status-badge');

  function setZoom(val) {
    val = Math.max(1, Math.min(5, parseFloat(val)));
    slider.value = val;
    zoomVal.textContent = val.toFixed(1) + 'x';
    fetch('/set_zoom', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({zoom: val})
    });
  }

  function changeZoom(delta) {
    setZoom(parseFloat(slider.value) + delta);
  }

  function takeSnapshot() {
    window.location.href = '/snapshot';
  }

  // Frame polling via fetch() to read headers + image
  const feed = document.getElementById('feed');
  let fetching = false;
  function fetchFrame() {
    if (fetching) return;
    fetching = true;
    fetch('/frame?' + Date.now())
      .then(r => {
        // Read detection state from headers
        const isHuman = r.headers.get('X-Human') === '1';
        const count = r.headers.get('X-Count') || '0';
        const centered = r.headers.get('X-Centered') === '1';
        const direction = r.headers.get('X-Direction') || '';

        if (isHuman) {
          badge.textContent = 'Human Detected (' + count + ')';
          badge.classList.add('alert');
        } else {
          badge.textContent = 'Monitoring';
          badge.classList.remove('alert');
        }
        // Sound: gunshot if centered, direction tones if off-center, silence if no human
        if (isHuman && centered) {
          updateSound(true, '');
        } else if (isHuman && direction) {
          updateSound(false, direction);
        } else {
          updateSound(false, '');
        }

        return r.blob();
      })
      .then(blob => {
        const url = URL.createObjectURL(blob);
        feed.onload = () => URL.revokeObjectURL(url);
        feed.src = url;
        fetching = false;
        setTimeout(fetchFrame, 100);
      })
      .catch(() => {
        fetching = false;
        setTimeout(fetchFrame, 500);
      });
  }
  fetchFrame();

  // --- Sound system (all Web Audio API, no speechSynthesis) ---
  let audioCtx = null;
  let soundEnabled = false;
  let shotTimer = null;
  let dirTimer = null;
  let currentSound = '';  // 'shot' or direction string
  const soundBtn = document.getElementById('sound-btn');

  function enableSound() {
    if (soundEnabled) {
      soundEnabled = false;
      clearTimers();
      soundBtn.textContent = 'Enable Sound';
      soundBtn.classList.remove('enabled');
      return;
    }
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    audioCtx.resume();
    // Unlock audio and load voice clips
    playTone(440, 0.1, 0.3);
    loadVoiceClips();
    soundEnabled = true;
    soundBtn.textContent = 'Sound ON';
    soundBtn.classList.add('enabled');
  }

  function clearTimers() {
    if (shotTimer) { clearInterval(shotTimer); shotTimer = null; }
    if (dirTimer) { clearInterval(dirTimer); dirTimer = null; }
    currentSound = '';
  }

  function playTone(freq, duration, vol) {
    if (!audioCtx) return;
    audioCtx.resume();
    const t = audioCtx.currentTime;
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.frequency.value = freq;
    osc.type = 'square';
    gain.gain.setValueAtTime(vol || 0.5, t);
    gain.gain.exponentialRampToValueAtTime(0.01, t + duration);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start(t);
    osc.stop(t + duration);
  }

  function playGunshot() {
    if (!audioCtx) return;
    audioCtx.resume();
    const ctx = audioCtx;
    const t = ctx.currentTime;
    const len = Math.floor(ctx.sampleRate * 0.15);
    const buf = ctx.createBuffer(1, len, ctx.sampleRate);
    const data = buf.getChannelData(0);
    for (let i = 0; i < len; i++) {
      data[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, 3);
    }
    const noise = ctx.createBufferSource();
    noise.buffer = buf;
    const filter = ctx.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.setValueAtTime(3000, t);
    filter.frequency.exponentialRampToValueAtTime(300, t + 0.1);
    const gain = ctx.createGain();
    gain.gain.setValueAtTime(2.0, t);
    gain.gain.exponentialRampToValueAtTime(0.01, t + 0.15);
    noise.connect(filter);
    filter.connect(gain);
    gain.connect(ctx.destination);
    noise.start(t);
    noise.stop(t + 0.15);
  }

  // Voice direction - use fetch to load WAV as ArrayBuffer, decode via AudioContext
  const voiceBuffers = {};
  function loadVoiceClips() {
    ['up','down','left','right'].forEach(d => {
      fetch('/sounds/' + d + '.wav')
        .then(r => r.arrayBuffer())
        .then(buf => audioCtx.decodeAudioData(buf))
        .then(decoded => { voiceBuffers[d] = decoded; })
        .catch(() => {});
    });
  }

  function playVoice(word) {
    if (!audioCtx || !voiceBuffers[word]) return;
    audioCtx.resume();
    const src = audioCtx.createBufferSource();
    src.buffer = voiceBuffers[word];
    src.connect(audioCtx.destination);
    src.start();
  }

  function playDirectionSound(dir) {
    if (!audioCtx) return;
    const words = dir.split(' ');
    words.forEach((w, i) => {
      setTimeout(() => playVoice(w), i * 400);
    });
  }

  // --- Flywheel toggle ---
  const flywheelBtn = document.getElementById('flywheel-btn');
  let flywheelActive = false;
  function toggleFlywheel() {
    flywheelActive = !flywheelActive;
    flywheelBtn.classList.toggle('on', flywheelActive);
    flywheelBtn.textContent = flywheelActive ? 'FLYWHEEL ON' : 'FLYWHEEL';
    fetch('/flywheel', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({state: flywheelActive})
    });
  }

  // --- Trigger (hold to fire) ---
  const triggerBtn = document.getElementById('trigger-btn');
  let triggerActive = false;
  function triggerStart(e) {
    if (e) e.preventDefault();
    if (triggerActive) return;
    triggerActive = true;
    triggerBtn.classList.add('on');
    fetch('/trigger', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({state: true})
    });
  }
  function triggerStop() {
    if (!triggerActive) return;
    triggerActive = false;
    triggerBtn.classList.remove('on');
    fetch('/trigger', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({state: false})
    });
  }

  // --- Joystick control (servo 0 = L/R, servo 2 = U/D) ---
  const joystick = document.getElementById('joystick');
  const knob = document.getElementById('joystick-knob');
  let joyActive = false;
  let joyRect = null;
  // Servo current and target angles (90 = center)
  let servoX = 90, servoY = 90;
  let targetX = 90, targetY = 90;

  // Smoothly interpolate servos toward target (lerp)
  const LERP_FACTOR = 0.08;  // blend 8% toward target each tick
  setInterval(() => {
    const prevX = servoX, prevY = servoY;
    servoX += (targetX - servoX) * LERP_FACTOR;
    servoY += (targetY - servoY) * LERP_FACTOR;
    // Only send if moved enough to matter (> 0.3 degrees)
    if (Math.abs(servoX - prevX) > 0.3 || Math.abs(servoY - prevY) > 0.3) {
      sendServo(0, Math.round(servoX));
      sendServo(2, Math.round(servoY));
    }
  }, 30);

  function joyUpdate(clientX, clientY) {
    if (!joyRect) joyRect = joystick.getBoundingClientRect();
    const r = joyRect.width / 2;
    let dx = clientX - (joyRect.left + r);
    let dy = clientY - (joyRect.top + r);
    // Clamp to circle
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (dist > r) { dx = dx / dist * r; dy = dy / dist * r; }
    // Position knob
    knob.style.left = (r + dx) + 'px';
    knob.style.top = (r + dy) + 'px';
    // Map to target servo angles (full range)
    targetX = Math.round(90 - (dx / r) * 90);
    targetY = Math.round(90 - (dy / r) * 90);
    targetX = Math.max(0, Math.min(180, targetX));
    targetY = Math.max(40, Math.min(100, targetY));
  }

  let servoThrottle = {};
  function sendServo(ch, angle) {
    if (servoThrottle[ch]) return;
    servoThrottle[ch] = true;
    fetch('/servo', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({channel: ch, action: 'set', angle: angle})
    }).finally(() => {
      setTimeout(() => { servoThrottle[ch] = false; }, 50);
    });
  }

  function joyEnd() {
    joyActive = false;
    // Keep knob and servos at current position
  }

  joystick.addEventListener('mousedown', (e) => {
    joyActive = true;
    joyRect = joystick.getBoundingClientRect();
    joyUpdate(e.clientX, e.clientY);
  });
  document.addEventListener('mousemove', (e) => {
    if (joyActive) joyUpdate(e.clientX, e.clientY);
  });
  document.addEventListener('mouseup', () => {
    if (joyActive) joyEnd();
  });
  joystick.addEventListener('touchstart', (e) => {
    e.preventDefault();
    joyActive = true;
    joyRect = joystick.getBoundingClientRect();
    joyUpdate(e.touches[0].clientX, e.touches[0].clientY);
  });
  document.addEventListener('touchmove', (e) => {
    if (joyActive) {
      e.preventDefault();
      joyUpdate(e.touches[0].clientX, e.touches[0].clientY);
    }
  }, {passive: false});
  document.addEventListener('touchend', () => {
    if (joyActive) joyEnd();
  });

  function updateSound(centered, direction) {
    if (!soundEnabled) return;
    if (centered) {
      if (currentSound !== 'shot') {
        clearTimers();
        currentSound = 'shot';
        playGunshot();
        shotTimer = setInterval(playGunshot, 350);
      }
    } else if (direction) {
      if (currentSound !== 'dir:' + direction) {
        clearTimers();
        currentSound = 'dir:' + direction;
        playDirectionSound(direction);
        dirTimer = setInterval(() => playDirectionSound(direction), 1000);
      }
    } else {
      if (currentSound) clearTimers();
    }
  }
</script>
</body>
</html>
"""

# Start background threads before Flask
cam_thread = threading.Thread(target=camera_reader, daemon=True)
cam_thread.start()
det_thread = threading.Thread(target=detection_worker, daemon=True)
det_thread.start()
servo_thread = threading.Thread(target=servo_sweep_worker, daemon=True)
servo_thread.start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=927, threaded=True)
