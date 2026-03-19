#!/usr/bin/env python3
"""Thin hardware server for Raspberry Pi — camera, servos, relays.

Detection and UI are offloaded to a remote machine.
"""

import threading
import time
import os
import io
import subprocess
from flask import Flask, Response, jsonify, request, send_file
import cv2
import numpy as np
from gpiozero import OutputDevice
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo as adafruit_servo
import board

app = Flask(__name__)

# --- Flywheel relay on BCM GPIO 17 (BOARD pin 11) ---
_flywheel_relay = OutputDevice(17, active_high=False, initial_value=False)
flywheel_on = False

# --- Trigger relay on BCM GPIO 27 (BOARD pin 13) ---
_trigger_relay = OutputDevice(27, active_high=False, initial_value=False)
trigger_on = False

# --- PCA9685 Servos on channels 0 and 2 ---
# Bypass ServoKit/PCA9685 reset to prevent servo jerk on restart/reboot.
_pca = None
_servo_objs = {}
try:
    i2c = board.I2C()
    _orig_reset = PCA9685.reset
    PCA9685.reset = lambda self: None
    _pca = PCA9685(i2c, address=0x40)
    PCA9685.reset = _orig_reset
    current_prescale = _pca.prescale_reg
    if current_prescale != 121:
        _pca.frequency = 50
    else:
        _pca.mode1_reg = _pca.mode1_reg | 0xA0
    _servo_objs[0] = adafruit_servo.Servo(_pca.channels[0])
    _servo_objs[2] = adafruit_servo.Servo(_pca.channels[2])
except (ValueError, OSError) as e:
    print(f"WARNING: PCA9685 servo controller not found ({e}). Servo disabled.")
    _pca = None
servo_angles = {0: 90.0, 2: 90.0}
servo_targets = {0: 90.0, 2: 90.0}
_servo_lock = threading.Lock()

# --- Shared state ---
zoom_level = 1.0
focus_value = 150
cam_ref = None
latest_frame = None
frame_lock = threading.Lock()
sweep_active = False


def usb_reset_camera():
    """Reset the Arducam by kernel-level USB deauthorize/reauthorize."""
    USB_AUTH = '/sys/bus/usb/devices/1-1.4/authorized'
    try:
        with open(USB_AUTH, 'w') as f:
            f.write('0')
        time.sleep(2)
        with open(USB_AUTH, 'w') as f:
            f.write('1')
        time.sleep(3)
        try:
            with open('/sys/module/uvcvideo/parameters/quirks', 'w') as f:
                f.write('0')
        except OSError:
            pass
        print("USB kernel reset completed.")
    except Exception as e:
        print(f"USB kernel reset failed: {e}")
        try:
            subprocess.run(['usbreset', 'Arducam_12MP'], timeout=10, capture_output=True)
            time.sleep(3)
            print("Fallback usbreset completed.")
        except Exception as e2:
            print(f"Fallback usbreset also failed: {e2}")


def open_camera():
    """Try to open a working USB camera, return the VideoCapture or None."""
    for idx in range(5):
        c = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if c.isOpened():
            c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            c.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            c.set(cv2.CAP_PROP_FPS, 30)
            c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            c.set(cv2.CAP_PROP_SHARPNESS, 5)
            result = [False]
            def try_read():
                ret, _ = c.read()
                result[0] = ret
            t = threading.Thread(target=try_read, daemon=True)
            t.start()
            t.join(timeout=5)
            if t.is_alive():
                print(f"Camera {idx} blocked on read (stuck firmware), releasing...")
                c.release()
                continue
            if result[0]:
                return c
            c.release()
    return None


def camera_reader():
    """Dedicated thread: reads frames as fast as possible, keeps only the latest."""
    global latest_frame, cam_ref
    MAX_FAILURES = 30
    consecutive_open_failures = 0

    while True:
        cap = open_camera()
        if cap is None:
            consecutive_open_failures += 1
            if consecutive_open_failures >= 2:
                print("Camera stuck after 2 attempts, trying USB power cycle...")
                usb_reset_camera()
                consecutive_open_failures = 0
            else:
                print(f"WARNING: No working camera found (attempt {consecutive_open_failures}), retrying in 3s...")
                time.sleep(3)
            continue
        consecutive_open_failures = 0

        cam_ref = cap
        print("Camera opened successfully.")
        fail_count = 0
        read_timeout_count = 0
        while True:
            result = [False, None]
            def do_read():
                ret, frame = cap.read()
                result[0] = ret
                result[1] = frame
            t = threading.Thread(target=do_read, daemon=True)
            t.start()
            t.join(timeout=3)
            if t.is_alive():
                read_timeout_count += 1
                print(f"Camera read blocked (timeout {read_timeout_count})...")
                if read_timeout_count >= 3:
                    print("Camera firmware stuck, forcing USB power cycle...")
                    cap.release()
                    usb_reset_camera()
                    break
                continue
            read_timeout_count = 0
            ret, frame = result
            if ret:
                fail_count = 0
                frame = cv2.flip(frame, -1)
                with frame_lock:
                    latest_frame = frame
            else:
                fail_count += 1
                if fail_count >= MAX_FAILURES:
                    print(f"Camera failed {MAX_FAILURES} consecutive reads, reconnecting...")
                    break
                time.sleep(0.03)

        cap.release()
        time.sleep(1)


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


def servo_smooth_worker():
    """Server-side smooth interpolation — moves servos in tiny steps at 100Hz."""
    MAX_STEP = 0.3
    IDLE_TICKS = 50
    idle_count = {0: 0, 2: 0}
    servo_released = {0: False, 2: False}
    while True:
        if _pca is None:
            time.sleep(1)
            continue
        with _servo_lock:
            for ch in (0, 2):
                current = servo_angles[ch]
                target = servo_targets[ch]
                diff = target - current
                if abs(diff) < 0.02:
                    idle_count[ch] += 1
                    if idle_count[ch] >= IDLE_TICKS and not servo_released[ch]:
                        try:
                            _servo_objs[ch].angle = None
                        except Exception:
                            pass
                        servo_released[ch] = True
                    continue
                idle_count[ch] = 0
                if servo_released[ch]:
                    servo_released[ch] = False
                step = max(-MAX_STEP, min(MAX_STEP, diff))
                new_angle = current + step
                new_angle = max(0.0, min(180.0, new_angle))
                try:
                    _servo_objs[ch].angle = new_angle
                except Exception:
                    pass
                servo_angles[ch] = new_angle
        time.sleep(0.01)


def sweep_worker():
    """Continuously sweeps servo targets back and forth."""
    global sweep_active
    LIMITS = {0: (0.0, 180.0), 2: (40.0, 100.0)}
    SWEEP_STEP = 0.5
    RETURN_STEP = 0.5
    CENTER = 90.0
    direction = {0: 1, 2: 1}

    while True:
        if _pca is None:
            time.sleep(0.1)
            continue

        if sweep_active:
            with _servo_lock:
                for ch in (0, 2):
                    lo, hi = LIMITS[ch]
                    current_target = servo_targets[ch]
                    if current_target >= hi:
                        direction[ch] = -1
                    elif current_target <= lo:
                        direction[ch] = 1
                    servo_targets[ch] = current_target + direction[ch] * SWEEP_STEP
        else:
            with _servo_lock:
                all_centered = True
                for ch in (0, 2):
                    current_target = servo_targets[ch]
                    diff = CENTER - current_target
                    if abs(diff) < 0.1:
                        servo_targets[ch] = CENTER
                    else:
                        all_centered = False
                        step = max(-RETURN_STEP, min(RETURN_STEP, diff))
                        servo_targets[ch] = current_target + step
            if all_centered:
                time.sleep(0.1)
                continue

        time.sleep(0.05)


def generate_raw_frames():
    """Generator: yields raw MJPEG frames (zoomed, no overlays)."""
    while True:
        with frame_lock:
            frame = latest_frame
        if frame is None:
            time.sleep(0.03)
            continue
        frame = apply_zoom(frame.copy(), zoom_level)
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.033)


# --- Routes ---

@app.route('/raw_frame')
def raw_frame():
    """Return a single raw JPEG frame (zoomed, no overlays)."""
    with frame_lock:
        frame = latest_frame
    if frame is None:
        return "No frame", 503
    frame = apply_zoom(frame.copy(), zoom_level)
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(buf.tobytes(), mimetype='image/jpeg',
                    headers={'Cache-Control': 'no-store'})


@app.route('/video_feed')
def video_feed():
    """Raw MJPEG stream (for debugging or direct use)."""
    return Response(generate_raw_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


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


@app.route('/set_zoom', methods=['POST'])
def set_zoom():
    global zoom_level
    data = request.get_json(force=True)
    zoom_level = max(1.0, min(5.0, float(data.get('zoom', 1.0))))
    return jsonify(zoom=zoom_level)


@app.route('/set_focus', methods=['POST'])
def set_focus():
    global focus_value
    data = request.get_json(force=True)
    focus_value = max(1, min(1023, int(data.get('focus', 150))))
    subprocess.run(['/usr/bin/v4l2-ctl', '-d', '/dev/video0',
                    '--set-ctrl=focus_automatic_continuous=0',
                    f'--set-ctrl=focus_absolute={focus_value}'],
                   timeout=2)
    return jsonify(focus=focus_value)


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
    if _pca is None:
        return jsonify(angle=0, error='Servo controller not connected')
    angle = max(0, min(180, float(data.get('angle', 0))))
    with _servo_lock:
        servo_targets[channel] = angle
    return jsonify(channel=channel, angle=angle)


@app.route('/sweep', methods=['POST'])
def sweep_control():
    global sweep_active
    data = request.get_json(force=True)
    state = bool(data.get('state', False))
    sweep_active = state
    return jsonify(sweep=sweep_active)


@app.route('/status')
def status():
    return jsonify(
        zoom=zoom_level,
        flywheel=flywheel_on,
        trigger=trigger_on,
        sweep=sweep_active,
        servo_angles=servo_angles
    )


# Start background threads
threading.Thread(target=camera_reader, daemon=True).start()
threading.Thread(target=servo_smooth_worker, daemon=True).start()
threading.Thread(target=sweep_worker, daemon=True).start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=927, threaded=True)
