#!/usr/bin/env python3
"""Remote processing and UI server for Rober.

Runs on a more powerful machine. Fetches raw frames from the Pi hardware server,
runs MobileNet SSD detection, and serves the web UI.
"""

import threading
import time
import os
import io
import argparse
from flask import Flask, Response, render_template_string, jsonify, request, send_file
import cv2
import numpy as np
import requests as http_requests

app = Flask(__name__)

# --- Configuration ---
PI_URL = os.environ.get('PI_URL', 'http://192.168.1.105:927')

CONFIDENCE_THRESHOLD = 0.4
PERSON_CLASS = 15
DETECTION_HOLD_FRAMES = 15

# --- Load model ---
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
net = cv2.dnn.readNetFromCaffe(
    os.path.join(MODEL_DIR, 'MobileNetSSD_deploy.prototxt'),
    os.path.join(MODEL_DIR, 'MobileNetSSD_deploy.caffemodel')
)

# --- Shared state ---
latest_frame = None
frame_lock = threading.Lock()
detection_boxes = []
detection_lock = threading.Lock()
human_detected = False
human_count = 0
target_centered = False
aim_direction = ""


def frame_fetcher():
    """Continuously fetch raw frames from the Pi hardware server."""
    global latest_frame
    session = http_requests.Session()
    while True:
        try:
            r = session.get(f'{PI_URL}/raw_frame', timeout=3)
            if r.status_code == 200:
                arr = np.frombuffer(r.content, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    with frame_lock:
                        latest_frame = frame
            time.sleep(0.03)
        except Exception:
            time.sleep(1)


def detection_worker():
    """Runs person detection on latest frame."""
    global detection_boxes, human_detected, human_count, target_centered, aim_direction
    hold = 0

    while True:
        with frame_lock:
            frame = latest_frame

        if frame is None:
            time.sleep(0.05)
            continue

        # Frames from Pi are already zoomed
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
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

        centered = False
        direction = ""
        cx, cy = w // 2, h // 2
        thresh_x, thresh_y = w // 5, h // 5
        if len(boxes) > 0:
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

        time.sleep(0.03)


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
    """Generator that yields MJPEG frames with detection overlays."""
    while True:
        with frame_lock:
            frame = latest_frame

        if frame is None:
            time.sleep(0.03)
            continue

        frame = frame.copy()

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

        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.033)


# --- Helper for proxying POST requests to Pi ---
def proxy_post(endpoint):
    """Forward a POST request to the Pi and return its response."""
    try:
        r = http_requests.post(
            f'{PI_URL}{endpoint}',
            json=request.get_json(force=True),
            timeout=3
        )
        return (r.content, r.status_code,
                {'Content-Type': r.headers.get('Content-Type', 'application/json')})
    except http_requests.RequestException as e:
        return jsonify(error=f'Pi unreachable: {e}'), 502


# --- Routes ---

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
    frame = frame.copy()

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

    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
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
    sounds_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sounds')
    return send_file(os.path.join(sounds_dir, filename), mimetype='audio/wav')


@app.route('/set_zoom', methods=['POST'])
def set_zoom():
    return proxy_post('/set_zoom')


@app.route('/set_focus', methods=['POST'])
def set_focus():
    return proxy_post('/set_focus')


@app.route('/servo', methods=['POST'])
def servo_control():
    return proxy_post('/servo')


@app.route('/sweep', methods=['POST'])
def sweep_control():
    return proxy_post('/sweep')


@app.route('/flywheel', methods=['POST'])
def flywheel_control():
    return proxy_post('/flywheel')


@app.route('/trigger', methods=['POST'])
def trigger_control():
    return proxy_post('/trigger')


@app.route('/snapshot')
def snapshot():
    try:
        r = http_requests.get(f'{PI_URL}/snapshot', timeout=5)
        return Response(r.content, mimetype='image/jpeg',
                        headers={
                            'Content-Disposition': r.headers.get('Content-Disposition', ''),
                        })
    except http_requests.RequestException as e:
        return f'Pi unreachable: {e}', 502


@app.route('/status')
def status():
    try:
        pi_status = http_requests.get(f'{PI_URL}/status', timeout=3).json()
    except Exception:
        pi_status = {}
    pi_status.update(
        human_detected=human_detected,
        human_count=human_count,
        target_centered=target_centered,
    )
    return jsonify(pi_status)


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
  .btn-sweep {
    background: linear-gradient(135deg, #ff8800, #ffcc00); color: #fff;
    font-size: 1rem; padding: 10px 22px;
  }
  .btn-sweep.on { background: linear-gradient(135deg, #ffcc00, #ff8800); box-shadow: 0 0 16px rgba(255,200,0,0.5); }
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
    <button class="btn btn-sweep" id="sweep-btn" onclick="toggleSweep()">SWEEP</button>
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
    <div class="zoom-row">
      <label>Focus</label>
      <input type="range" id="focus-slider" min="1" max="1023" step="1" value="150"
             oninput="setFocus(this.value)">
      <span class="zoom-val" id="focus-val">150</span>
    </div>
    <button class="btn btn-snap" onclick="takeSnapshot()">Snapshot</button>
    <button class="btn btn-sound" id="sound-btn" onclick="enableSound()">Enable Sound</button>
  </div>

<script>
  const slider = document.getElementById('zoom-slider');
  const zoomVal = document.getElementById('zoom-val');
  const badge = document.getElementById('status-badge');

  let focusTimer = null;
  function setFocus(val) {
    val = Math.max(1, Math.min(1023, parseInt(val)));
    document.getElementById('focus-slider').value = val;
    document.getElementById('focus-val').textContent = val;
    if (focusTimer) clearTimeout(focusTimer);
    focusTimer = setTimeout(() => {
      fetch('/set_focus', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({focus: val})
      });
    }, 150);
  }

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

  // --- Sound system ---
  let audioCtx = null;
  let soundEnabled = false;
  let shotTimer = null;
  let dirTimer = null;
  let currentSound = '';
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

  // --- Sweep toggle ---
  const sweepBtn = document.getElementById('sweep-btn');
  let sweepActive = false;
  function toggleSweep() {
    sweepActive = !sweepActive;
    sweepBtn.classList.toggle('on', sweepActive);
    sweepBtn.textContent = sweepActive ? 'SWEEP ON' : 'SWEEP';
    fetch('/sweep', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({state: sweepActive})
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

  // --- Joystick control ---
  const joystick = document.getElementById('joystick');
  const knob = document.getElementById('joystick-knob');
  let joyActive = false;
  let joyRect = null;
  let targetX = 90, targetY = 90;
  let lastSentX = 90, lastSentY = 90;

  setInterval(() => {
    if (Math.abs(targetX - lastSentX) > 0.5 || Math.abs(targetY - lastSentY) > 0.5) {
      sendServo(0, targetX);
      sendServo(2, targetY);
      lastSentX = targetX;
      lastSentY = targetY;
    }
  }, 100);

  function joyUpdate(clientX, clientY) {
    if (!joyRect) joyRect = joystick.getBoundingClientRect();
    const r = joyRect.width / 2;
    let dx = clientX - (joyRect.left + r);
    let dy = clientY - (joyRect.top + r);
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (dist > r) { dx = dx / dist * r; dy = dy / dist * r; }
    knob.style.left = (r + dx) + 'px';
    knob.style.top = (r + dy) + 'px';
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
      setTimeout(() => { servoThrottle[ch] = false; }, 16);
    });
  }

  function joyEnd() {
    joyActive = false;
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

# Start background threads
threading.Thread(target=frame_fetcher, daemon=True).start()
threading.Thread(target=detection_worker, daemon=True).start()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rober remote processing server')
    parser.add_argument('--pi-url', default=os.environ.get('PI_URL', 'http://192.168.1.105:927'),
                        help='URL of the Pi hardware server')
    parser.add_argument('--port', type=int, default=5000, help='Port to listen on')
    args = parser.parse_args()
    PI_URL = args.pi_url
    app.run(host='0.0.0.0', port=args.port, threaded=True)
