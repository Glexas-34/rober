#!/usr/bin/env python3
"""USB Camera Web Interface for Raspberry Pi with human detection."""

import threading
import time
import os
import io
import json
import subprocess
from flask import Flask, Response, render_template_string, jsonify, request, send_file
import cv2
import numpy as np
from gpiozero import OutputDevice
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo as adafruit_servo
import board

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
# IMPORTANT: We bypass ServoKit and PCA9685's built-in reset() to prevent servos
# from jerking on service restart / reboot. PCA9685.__init__ calls self.reset()
# which writes 0x00 to MODE1, and the frequency setter briefly sleeps the chip.
# Both of these interrupt PWM output and cause brusque servo movements.
# Instead, we suppress reset() during init and only set frequency if prescale
# is wrong. No angle is commanded — servos stay where they physically are.
_pca = None
_servo_objs = {}
try:
    i2c = board.I2C()
    # Suppress reset() during __init__ to avoid disrupting PWM output
    _orig_reset = PCA9685.reset
    PCA9685.reset = lambda self: None
    _pca = PCA9685(i2c, address=0x40)
    PCA9685.reset = _orig_reset  # restore immediately
    # Only set frequency if prescale is not already 50Hz (prescale=121 for 25MHz/50Hz)
    current_prescale = _pca.prescale_reg
    if current_prescale != 121:
        _pca.frequency = 50
    else:
        # Already at 50Hz — just ensure auto-increment and oscillator are enabled
        # without the sleep/wake cycle that kills PWM
        _pca.mode1_reg = _pca.mode1_reg | 0xA0
    # Create servo objects on channels 0 and 2 — does NOT command any angle
    _servo_objs[0] = adafruit_servo.Servo(_pca.channels[0])
    _servo_objs[2] = adafruit_servo.Servo(_pca.channels[2])
except (ValueError, OSError) as e:
    print(f"WARNING: PCA9685 servo controller not found ({e}). Servo disabled.")
    _pca = None
servo_angles = {0: 88.0, 2: 67.0}
servo_targets = {0: 88.0, 2: 67.0}
_servo_lock = threading.Lock()

# --- Configuration ---
CONFIDENCE_THRESHOLD = 0.4
PERSON_CLASS = 15  # "person" in MobileNet SSD VOC classes
DETECTION_HOLD_FRAMES = 15

# --- Calibration: where the gun actually points (normalized 0-1, default=center) ---
CALIBRATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calibration.json')
calibration = {'x': 0.5, 'y': 0.5}  # normalized coords within frame
try:
    with open(CALIBRATION_FILE, 'r') as _cf:
        _cal_data = json.load(_cf)
        calibration['x'] = float(_cal_data.get('x', 0.5))
        calibration['y'] = float(_cal_data.get('y', 0.5))
    print(f"Calibration loaded: x={calibration['x']:.3f}, y={calibration['y']:.3f}")
except (FileNotFoundError, json.JSONDecodeError, ValueError):
    print("No calibration file found, using center default.")

# --- Load model ---
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
net = cv2.dnn.readNetFromCaffe(
    os.path.join(MODEL_DIR, 'MobileNetSSD_deploy.prototxt'),
    os.path.join(MODEL_DIR, 'MobileNetSSD_deploy.caffemodel')
)

# --- Shared state ---
zoom_level = 1.0
focus_value = 150
cam_ref = None  # reference to active camera for focus control
latest_frame = None
frame_lock = threading.Lock()
detection_boxes = []
detection_lock = threading.Lock()
human_detected = False
human_count = 0
target_centered = False
aim_direction = ""  # e.g. "left", "up right", etc.
sweep_active = False
track_shoot_active = False
track_dx = 0  # pixel offset of target from center (positive = target is right)
track_dy = 0  # pixel offset of target from center (positive = target is down)
track_frame_w = 1  # frame width for normalizing offsets
track_frame_h = 1  # frame height for normalizing offsets



def usb_reset_camera():
    """Reset the Arducam by kernel-level USB deauthorize/reauthorize.

    uhubctl does not work on this Pi 3 ('No compatible devices detected'),
    so we use the sysfs authorized flag which forces the kernel to fully
    disconnect and re-enumerate the USB device — equivalent to a physical
    unplug/replug cycle.
    """
    USB_AUTH = '/sys/bus/usb/devices/1-1.4/authorized'
    try:
        # Deauthorize (disconnect)
        with open(USB_AUTH, 'w') as f:
            f.write('0')
        time.sleep(2)
        # Reauthorize (re-enumerate)
        with open(USB_AUTH, 'w') as f:
            f.write('1')
        time.sleep(3)
        # Reset UVC quirks to sane default (0xFFFFFFFF has been seen and breaks streaming)
        try:
            with open('/sys/module/uvcvideo/parameters/quirks', 'w') as f:
                f.write('0')
        except OSError:
            pass
        print("USB kernel reset completed.")
    except Exception as e:
        print(f"USB kernel reset failed: {e}")
        # Fallback: try usbreset utility
        try:
            subprocess.run(['usbreset', 'Arducam_12MP'], timeout=10, capture_output=True)
            time.sleep(3)
            print("Fallback usbreset completed.")
        except Exception as e2:
            print(f"Fallback usbreset also failed: {e2}")


def open_camera():
    """Try to open a working USB camera, return the VideoCapture or None.
    Uses a timeout thread to avoid blocking forever on a stuck camera."""
    for idx in range(5):
        c = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if c.isOpened():
            c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            c.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            c.set(cv2.CAP_PROP_FPS, 30)
            c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            c.set(cv2.CAP_PROP_SHARPNESS, 5)
            # Use a thread to timeout the initial read (stuck cameras block here)
            result = [False]
            def try_read():
                ret, _ = c.read()
                result[0] = ret
            t = threading.Thread(target=try_read, daemon=True)
            t.start()
            t.join(timeout=5)  # 5 second timeout for first frame
            if t.is_alive():
                print(f"Camera {idx} blocked on read (stuck firmware), releasing...")
                c.release()
                continue
            if result[0]:
                return c
            c.release()
    return None


def camera_reader():
    """Dedicated thread: reads frames as fast as possible, keeps only the latest.
    Auto-recovers if the camera stops delivering frames."""
    global latest_frame, cam_ref
    MAX_FAILURES = 30  # consecutive failures before reconnect
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
            # Use a thread to timeout reads (stuck cameras block cap.read())
            result = [False, None]
            def do_read():
                ret, frame = cap.read()
                result[0] = ret
                result[1] = frame
            t = threading.Thread(target=do_read, daemon=True)
            t.start()
            t.join(timeout=3)  # 3 second timeout per frame read
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
                # Camera is mounted right-side up, no flip needed
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
    global track_dx, track_dy, track_frame_w, track_frame_h
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

        # Check if any target head is near the calibrated aim point
        centered = False
        direction = ""
        cx = int(calibration['x'] * w)
        cy = int(calibration['y'] * h)
        thresh_x, thresh_y = w * 4 // 75, h * 8 // 75
        raw_dx, raw_dy = 0, 0
        if len(boxes) > 0:
            # Use the first (most confident) detection
            bx, by, bw, bh = boxes[0]
            head_x = bx + bw // 2
            head_y = by + int(bh * 0.1)
            dx = head_x - cx
            dy = head_y - cy
            raw_dx, raw_dy = dx, dy
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
            track_dx = raw_dx
            track_dy = raw_dy
            track_frame_w = w
            track_frame_h = h
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


def _recover_pca9685():
    """Attempt to recover a PCA9685 that has stopped outputting PWM.

    Common causes: power glitch from relay switching causes chip brownout/reset,
    putting it back into default sleep mode (MODE1=0x11). Also handles I2C bus
    errors by re-initializing the bus.

    Returns True if recovery succeeded, False otherwise.
    """
    global _pca, _servo_objs
    try:
        mode1 = _pca.mode1_reg
        # Check if chip is asleep (bit 4 = SLEEP). Default after reset is 0x11.
        if mode1 & 0x10:
            print(f"PCA9685 recovery: chip was asleep (MODE1=0x{mode1:02X}), waking up...")
            # Clear SLEEP bit, keep AI (auto-increment)
            _pca.mode1_reg = (mode1 & ~0x10) | 0x20
            time.sleep(0.005)  # oscillator stabilization (500us min per datasheet)
            # Set RESTART bit to resume PWM from where it was
            _pca.mode1_reg = _pca.mode1_reg | 0x80
            print("PCA9685 recovery: chip woken, RESTART issued.")

        # Verify/fix prescale (should be 121 for 50Hz)
        prescale = _pca.prescale_reg
        if prescale != 121:
            print(f"PCA9685 recovery: prescale was {prescale}, resetting to 121 (50Hz)...")
            # Must sleep chip to change prescale
            cur = _pca.mode1_reg
            _pca.mode1_reg = (cur & ~0x80) | 0x10  # SLEEP=1, clear RESTART
            time.sleep(0.005)
            _pca.prescale_reg = 121
            _pca.mode1_reg = cur & ~0x10  # clear SLEEP
            time.sleep(0.005)
            _pca.mode1_reg = _pca.mode1_reg | 0x80  # RESTART

        # Ensure AI + oscillator enabled
        _pca.mode1_reg = _pca.mode1_reg | 0xA0
        print(f"PCA9685 recovery: OK (MODE1=0x{_pca.mode1_reg:02X}, prescale={_pca.prescale_reg})")
        return True

    except OSError as e:
        print(f"PCA9685 recovery: I2C still failing ({e}), attempting full re-init...")
        try:
            # Full re-init: new I2C bus + new PCA9685 object
            i2c = board.I2C()
            orig_reset = PCA9685.reset
            PCA9685.reset = lambda self: None
            _pca = PCA9685(i2c, address=0x40)
            PCA9685.reset = orig_reset
            if _pca.prescale_reg != 121:
                _pca.frequency = 50
            else:
                _pca.mode1_reg = _pca.mode1_reg | 0xA0
            _servo_objs[0] = adafruit_servo.Servo(_pca.channels[0])
            _servo_objs[2] = adafruit_servo.Servo(_pca.channels[2])
            print("PCA9685 recovery: full re-init succeeded.")
            return True
        except Exception as e2:
            print(f"PCA9685 recovery: full re-init failed ({e2}).")
            return False


def servo_smooth_worker():
    """Server-side smooth interpolation — moves servos in tiny steps at steady 100Hz.
    PWM is never released — servos hold position permanently.
    Includes automatic recovery if PCA9685 stops responding."""
    MAX_STEP = 0.3  # max degrees per tick (0.3 deg * 100Hz = 30 deg/sec)
    # On boot, servos have no PWM yet. We must not command any angle until the
    # user actually provides input (changes the target from its initial value).
    first_move_seen = {0: False, 2: False}
    last_written = {0: None, 2: None}
    consecutive_errors = 0
    RECOVERY_THRESHOLD = 5  # attempt recovery after this many consecutive I2C errors
    HEALTH_CHECK_INTERVAL = 5.0  # seconds between proactive health checks
    last_health_check = time.monotonic()

    while True:
        if _pca is None:
            time.sleep(1)
            continue

        now = time.monotonic()

        # Proactive health check: verify PCA9685 is still awake even when
        # no servo commands are being sent (catches silent brownout resets)
        if now - last_health_check >= HEALTH_CHECK_INTERVAL:
            last_health_check = now
            try:
                mode1 = _pca.mode1_reg
                if mode1 & 0x10:  # SLEEP bit set = chip reset itself
                    print(f"PCA9685 health check: chip asleep (MODE1=0x{mode1:02X}), recovering...")
                    if _recover_pca9685():
                        consecutive_errors = 0
                        # Force re-send of current angles to restore PWM output
                        last_written[0] = None
                        last_written[2] = None
            except OSError:
                consecutive_errors += 1

        with _servo_lock:
            for ch in (0, 2):
                current = servo_angles[ch]
                target = servo_targets[ch]
                diff = target - current
                if abs(diff) < 0.02 and last_written[ch] is not None:
                    continue
                # First time this servo is being moved since boot
                if not first_move_seen[ch]:
                    if abs(diff) < 0.02:
                        continue  # no user input yet, stay idle
                    first_move_seen[ch] = True
                    servo_angles[ch] = target
                    try:
                        hw_angle = (130.0 - target) if ch == 2 else target
                        _servo_objs[ch].angle = hw_angle
                        consecutive_errors = 0
                    except OSError as e:
                        consecutive_errors += 1
                        print(f"Servo ch{ch} write error ({consecutive_errors}): {e}")
                    last_written[ch] = target
                    continue
                step = max(-MAX_STEP, min(MAX_STEP, diff))
                new_angle = current + step
                new_angle = max(0.0, min(180.0, new_angle))
                rounded = round(new_angle, 1)
                if rounded != last_written[ch]:
                    try:
                        hw_angle = (130.0 - new_angle) if ch == 2 else new_angle
                        _servo_objs[ch].angle = hw_angle
                        consecutive_errors = 0
                    except OSError as e:
                        consecutive_errors += 1
                        if consecutive_errors % RECOVERY_THRESHOLD == 0:
                            print(f"Servo ch{ch}: {consecutive_errors} consecutive I2C errors, recovering...")
                            if _recover_pca9685():
                                consecutive_errors = 0
                                last_written[0] = None
                                last_written[2] = None
                                break  # restart the loop to re-send angles
                    last_written[ch] = rounded
                servo_angles[ch] = new_angle

        # Back off slightly when errors are happening to avoid hammering a dead bus
        if consecutive_errors > 0:
            time.sleep(0.05)
        else:
            time.sleep(0.01)  # 100Hz


def sweep_worker():
    """Continuously sweeps servo targets back and forth independently.
    When sweep is deactivated, smoothly returns targets to center once, then idles."""
    global sweep_active
    LIMITS = {0: (0.0, 180.0), 2: (30.0, 100.0)}
    SWEEP_STEP = 0.5
    RETURN_STEP = 0.5
    CENTER = {0: 88.0, 2: 67.0}
    direction = {0: 1, 2: 1}
    was_sweeping = False  # track whether we need to return to center

    while True:
        if _pca is None:
            time.sleep(0.1)
            continue

        if sweep_active:
            was_sweeping = True
            with _servo_lock:
                for ch in (0, 2):
                    lo, hi = LIMITS[ch]
                    current_target = servo_targets[ch]
                    if current_target >= hi:
                        direction[ch] = -1
                    elif current_target <= lo:
                        direction[ch] = 1
                    servo_targets[ch] = current_target + direction[ch] * SWEEP_STEP
        elif was_sweeping:
            # Sweep just stopped — smoothly return to center, then stop
            with _servo_lock:
                all_centered = True
                for ch in (0, 2):
                    current_target = servo_targets[ch]
                    diff = CENTER[ch] - current_target
                    if abs(diff) < RETURN_STEP:
                        servo_targets[ch] = CENTER[ch]
                    else:
                        all_centered = False
                        step = max(-RETURN_STEP, min(RETURN_STEP, diff))
                        servo_targets[ch] = current_target + step
            if all_centered:
                was_sweeping = False
        else:
            # Not sweeping, not returning — idle
            time.sleep(0.1)
            continue

        time.sleep(0.05)


def track_shoot_worker():
    """Track & Shoot: sweeps to find a person, tracks them to center,
    shoots for 2 seconds, then continues tracking.
    If person disappears for 10 seconds, resumes sweep."""
    global track_shoot_active, sweep_active, flywheel_on, trigger_on

    CENTER = {0: 88.0, 2: 67.0}
    TRACK_GAIN = 0.15  # how aggressively to chase (degrees per pixel-fraction)
    CENTERED_FRAMES_NEEDED = 3  # require N consecutive centered frames before shooting
    LOST_TIMEOUT = 10.0  # seconds without detection before resuming sweep
    centered_count = 0
    last_seen_time = None  # timestamp of last person detection

    while True:
        if not track_shoot_active:
            centered_count = 0
            last_seen_time = None
            time.sleep(0.1)
            continue

        with detection_lock:
            detected = human_detected
            centered = target_centered
            dx = track_dx
            dy = track_dy
            fw = track_frame_w
            fh = track_frame_h

        if not detected:
            centered_count = 0
            now = time.monotonic()
            if last_seen_time is None:
                # First loop or just activated — start sweep immediately
                if not sweep_active:
                    sweep_active = True
            elif now - last_seen_time >= LOST_TIMEOUT:
                # Person lost for 10 seconds — resume sweep
                if not sweep_active:
                    sweep_active = True
            # else: within 10s grace period, hold position (no sweep yet)
            time.sleep(0.05)
            continue

        # Person detected — record time, stop sweep and track
        last_seen_time = time.monotonic()
        if sweep_active:
            sweep_active = False

        if centered:
            centered_count += 1
        else:
            centered_count = 0
            # Adjust servos proportionally to offset
            # dx positive = target is to the right of frame = need to decrease ch0 angle
            # dy positive = target is below center = need to increase ch2 angle
            norm_dx = dx / (fw / 2)  # -1 to 1
            norm_dy = dy / (fh / 2)  # -1 to 1
            pan_adjust = -norm_dx * TRACK_GAIN * 90  # scale to degrees
            tilt_adjust = norm_dy * TRACK_GAIN * 35   # ch2 has smaller range
            with _servo_lock:
                servo_targets[0] = max(0.0, min(180.0, servo_targets[0] + pan_adjust))
                servo_targets[2] = max(30.0, min(100.0, servo_targets[2] + tilt_adjust))

        if centered_count >= CENTERED_FRAMES_NEEDED:
            # --- SHOOT sequence (2 seconds) ---
            centered_count = 0

            # Flywheel ON
            _flywheel_relay.on()
            flywheel_on = True
            time.sleep(0.5)  # let flywheel spin up

            if not track_shoot_active:
                _flywheel_relay.off()
                flywheel_on = False
                continue

            # Trigger ON
            _trigger_relay.on()
            trigger_on = True
            time.sleep(2.0)  # shoot for 2 seconds

            # Trigger OFF
            _trigger_relay.off()
            trigger_on = False
            time.sleep(0.5)

            # Flywheel OFF
            _flywheel_relay.off()
            flywheel_on = False

            # Continue tracking (loop back)
            continue

        time.sleep(0.05)


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


def draw_aim_zone(frame):
    """Draw the calibration crosshair and the 'centered' threshold rectangle on the frame."""
    h, w = frame.shape[:2]
    cx = int(calibration['x'] * w)
    cy = int(calibration['y'] * h)
    thresh_x = w * 4 // 75
    thresh_y = h * 8 // 75
    # Threshold rectangle (the "shoot" zone)
    cv2.rectangle(frame, (cx - thresh_x, cy - thresh_y), (cx + thresh_x, cy + thresh_y),
                  (0, 255, 255), 3)
    # Crosshair at calibration point
    arm = 16
    cv2.line(frame, (cx - arm, cy), (cx + arm, cy), (0, 255, 255), 2)
    cv2.line(frame, (cx, cy - arm), (cx, cy + arm), (0, 255, 255), 2)


# --- Pre-encoded stream frame (background thread encodes, generators just yield) ---
_stream_buf = None
_stream_lock = threading.Lock()
STREAM_W, STREAM_H = 640, 360  # downscaled for fast encode + transfer


def stream_encoder():
    """Background thread: encodes stream frames at reduced resolution."""
    global _stream_buf
    prev_frame_id = None
    while True:
        with frame_lock:
            frame = latest_frame
        if frame is None:
            time.sleep(0.01)
            continue
        # Skip if same frame object (no new camera read)
        fid = id(frame)
        if fid == prev_frame_id:
            time.sleep(0.005)
            continue
        prev_frame_id = fid

        frame = apply_zoom(frame.copy(), zoom_level)
        # Downscale for stream
        small = cv2.resize(frame, (STREAM_W, STREAM_H), interpolation=cv2.INTER_NEAREST)
        draw_aim_zone(small)

        with detection_lock:
            boxes = detection_boxes.copy()
            detected = human_detected
            count = human_count

        if detected and boxes:
            # Scale detection boxes to stream resolution
            fh, fw = frame.shape[:2]
            sx, sy = STREAM_W / fw, STREAM_H / fh
            for (x, y, w, h) in boxes:
                rx, ry, rw, rh = int(x*sx), int(y*sy), int(w*sx), int(h*sy)
                cv2.rectangle(small, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
                head_x = rx + rw // 2
                head_y = ry + int(rh * 0.1)
                r = max(12, rw // 6)
                draw_target(small, head_x, head_y, r)
            cv2.putText(small, f"HUMAN DETECTED ({count})", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(small, ts, (8, STREAM_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        if zoom_level > 1.0:
            cv2.putText(small, f"{zoom_level:.1f}x", (STREAM_W - 60, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        _, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 60])
        encoded = (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        with _stream_lock:
            _stream_buf = encoded


def generate_frames():
    """Generator that yields pre-encoded MJPEG frames."""
    prev = None
    while True:
        with _stream_lock:
            buf = _stream_buf
        if buf is None or buf is prev:
            time.sleep(0.005)
            continue
        prev = buf
        yield buf


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
    draw_aim_zone(frame)

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
    """Serve voice clip WAV files."""
    sounds_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sounds')
    return send_file(os.path.join(sounds_dir, filename), mimetype='audio/wav')


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
    # When deactivated, sweep_worker handles smooth return to center
    return jsonify(sweep=sweep_active)


@app.route('/track_shoot', methods=['POST'])
def track_shoot_control():
    global track_shoot_active, sweep_active, flywheel_on, trigger_on
    data = request.get_json(force=True)
    state = bool(data.get('state', False))
    track_shoot_active = state
    if not state:
        # Deactivating: stop sweep, stop flywheel/trigger, return to center
        sweep_active = False
        _flywheel_relay.off()
        flywheel_on = False
        _trigger_relay.off()
        trigger_on = False
        with _servo_lock:
            servo_targets[0] = 88.0
            servo_targets[2] = 67.0
    return jsonify(track_shoot=track_shoot_active)


@app.route('/reset', methods=['POST'])
def reset_all():
    """Stop sweep, flywheel, trigger, track_shoot, and return servos to center."""
    global sweep_active, flywheel_on, trigger_on, track_shoot_active
    sweep_active = False
    track_shoot_active = False
    _flywheel_relay.off()
    flywheel_on = False
    _trigger_relay.off()
    trigger_on = False
    with _servo_lock:
        servo_targets[0] = 88.0
        servo_targets[2] = 67.0
    return jsonify(ok=True)


@app.route('/calibration', methods=['GET'])
def get_calibration():
    return jsonify(x=calibration['x'], y=calibration['y'])


@app.route('/calibration', methods=['POST'])
def set_calibration():
    data = request.get_json(force=True)
    calibration['x'] = max(0.0, min(1.0, float(data.get('x', 0.5))))
    calibration['y'] = max(0.0, min(1.0, float(data.get('y', 0.5))))
    with open(CALIBRATION_FILE, 'w') as f:
        json.dump({'x': calibration['x'], 'y': calibration['y']}, f)
    print(f"Calibration saved: x={calibration['x']:.3f}, y={calibration['y']:.3f}")
    return jsonify(x=calibration['x'], y=calibration['y'])


@app.route('/status')
def status():
    return jsonify(
        zoom=zoom_level,
        human_detected=human_detected,
        human_count=human_count,
        target_centered=target_centered,
        aim_direction=aim_direction,
        flywheel=flywheel_on,
        trigger=trigger_on,
        track_shoot=track_shoot_active,
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
  .tab-bar {
    display: flex; background: rgba(0,0,0,0.3);
    border-bottom: 2px solid rgba(255,255,255,0.15);
  }
  .tab-btn {
    flex: 1; padding: 8px 0; text-align: center; cursor: pointer;
    font-size: 0.85rem; font-weight: 600; color: #aaa;
    background: transparent; border: none; border-bottom: 3px solid transparent;
    transition: color 0.2s, border-color 0.2s;
  }
  .tab-btn.active { color: #fff; border-bottom-color: #ff8800; }
  .tab-btn:hover { color: #ddd; }
  .tab-content { display: none; flex: 1; flex-direction: column; overflow: hidden; }
  .tab-content.active { display: flex; }
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
  .btn-reset {
    background: linear-gradient(135deg, #666, #444); color: #fff;
    font-size: 1rem; padding: 10px 22px; align-self: center;
  }
  .btn-shoot {
    background: linear-gradient(135deg, #cc0000, #880000); color: #fff;
    font-size: 1.1rem; padding: 14px 24px; align-self: center; font-weight: 700;
  }
  .btn-shoot.on { background: linear-gradient(135deg, #ff0000, #ff4400); box-shadow: 0 0 20px rgba(255,0,0,0.7); }
  .btn-track-shoot {
    background: linear-gradient(135deg, #ff4400, #cc0088); color: #fff;
    font-size: 1rem; padding: 10px 22px; font-weight: 700;
  }
  .btn-track-shoot.on { background: linear-gradient(135deg, #ff0044, #ff00cc); box-shadow: 0 0 20px rgba(255,0,136,0.7); }
  .joystick-container {
    position: relative; width: 150px; height: 150px;
    background: conic-gradient(#ff0000, #ff8800, #ffff00, #00cc00, #0088ff, #8800ff, #ff0000);
    border-radius: 8px; border: 3px solid rgba(255,255,255,0.3);
    touch-action: none; user-select: none;
  }
  .joystick-knob {
    position: absolute; width: 40px; height: 40px;
    background: radial-gradient(circle, #fff, #ddd);
    border: 2px solid rgba(0,0,0,0.3);
    border-radius: 50%;
    top: 47.1%; left: 51.1%; transform: translate(-50%, -50%);
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

  <div class="tab-bar">
    <button class="tab-btn active" onclick="switchTab('control')">Control</button>
    <button class="tab-btn" onclick="switchTab('calibration')">Calibration</button>
  </div>

  <div class="tab-content active" id="tab-control">
    <div class="feed-container" style="cursor:crosshair;">
      <img id="feed" alt="Live Feed">
    </div>

    <div class="controls">
      <button class="btn btn-flywheel" id="flywheel-btn" onclick="toggleFlywheel()">FLYWHEEL</button>
      <button class="btn btn-sweep" id="sweep-btn" onclick="toggleSweep()">SWEEP</button>
      <button class="btn btn-track-shoot" id="track-shoot-btn" onclick="toggleTrackShoot()">TRACK&amp;SHOOT</button>
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
        <button class="btn btn-reset" onclick="resetAll()">RESET</button>
        <button class="btn btn-shoot" id="shoot-btn"
                onmousedown="shootStart()" onmouseup="shootStop()" onmouseleave="shootStop()"
                ontouchstart="shootStart(event)" ontouchend="shootStop()" ontouchcancel="shootStop()">SHOOT</button>
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
  </div>

  <div class="tab-content" id="tab-calibration">
    <div class="feed-container" style="position:relative; cursor:crosshair;">
      <img id="cal-feed" alt="Calibration Feed" style="max-width:100%; max-height:100%; object-fit:contain;">
      <svg id="cal-overlay" style="position:absolute; top:0; left:0; width:100%; height:100%; pointer-events:none;"></svg>
    </div>
    <div class="controls" style="justify-content:center; gap:16px;">
      <div style="font-size:0.85rem; text-align:center;">
        <span style="animation: rainbow-text 3s linear infinite;">Tap where the gun is actually pointing</span><br>
        <span id="cal-coords" style="color:#ff8800; font-size:0.8rem;"></span>
      </div>
      <button class="btn btn-snap" id="cal-submit" onclick="submitCalibration()" style="opacity:0.4; pointer-events:none;">SAVE CALIBRATION</button>
      <button class="btn btn-reset" onclick="clearCalibration()">CLEAR</button>
    </div>
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

  // MJPEG stream for low-latency video (single persistent connection)
  const feed = document.getElementById('feed');
  feed.src = '/video_feed';

  // Lightweight status poll for detection badges and sound
  function pollStatus() {
    fetch('/status')
      .then(r => r.json())
      .then(data => {
        const isHuman = data.human_detected;
        const count = data.human_count;
        const centered = data.target_centered;
        const direction = data.aim_direction || '';
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
      })
      .catch(() => {});
  }
  setInterval(pollStatus, 200);

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

  // --- Reset all ---
  function resetAll() {
    fetch('/reset', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({})
    });
    // Update UI state to match
    sweepActive = false;
    sweepBtn.classList.remove('on');
    sweepBtn.textContent = 'SWEEP';
    flywheelActive = false;
    flywheelBtn.classList.remove('on');
    flywheelBtn.textContent = 'FLYWHEEL';
    triggerActive = false;
    triggerBtn.classList.remove('on');
    trackShootActive = false;
    trackShootBtn.classList.remove('on');
    trackShootBtn.textContent = 'TRACK&SHOOT';
    // Reset joystick knob to resting position (pan=88, tilt=67)
    targetX = 88; targetY = 67;
    lastSentX = 88; lastSentY = 67;
    const r = joystick.offsetWidth / 2;
    const dx = (90 - 88) / 90 * r;
    const dy = (65 - 67) / 35 * r;
    knob.style.left = (r + dx) + 'px';
    knob.style.top = (r + dy) + 'px';
    joyRect = null;
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

  // --- SHOOT button: flywheel on immediately, trigger after 0.5s delay ---
  const shootBtn = document.getElementById('shoot-btn');
  let shootActive = false;
  let shootTriggerTimer = null;
  let shootFlywheelOffTimer = null;
  function shootStart(e) {
    if (e) e.preventDefault();
    if (shootActive) return;
    shootActive = true;
    shootBtn.classList.add('on');
    // Turn on flywheel immediately
    if (shootFlywheelOffTimer) { clearTimeout(shootFlywheelOffTimer); shootFlywheelOffTimer = null; }
    fetch('/flywheel', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({state: true})
    });
    flywheelActive = true;
    flywheelBtn.classList.add('on');
    flywheelBtn.textContent = 'FLYWHEEL ON';
    // Turn on trigger after 0.5s
    shootTriggerTimer = setTimeout(() => {
      fetch('/trigger', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({state: true})
      });
      triggerActive = true;
      triggerBtn.classList.add('on');
    }, 500);
  }
  function shootStop() {
    if (!shootActive) return;
    shootActive = false;
    shootBtn.classList.remove('on');
    // Cancel trigger timer if not yet fired
    if (shootTriggerTimer) { clearTimeout(shootTriggerTimer); shootTriggerTimer = null; }
    // Turn off trigger immediately
    fetch('/trigger', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({state: false})
    });
    triggerActive = false;
    triggerBtn.classList.remove('on');
    // Turn off flywheel after 0.5s
    shootFlywheelOffTimer = setTimeout(() => {
      fetch('/flywheel', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({state: false})
      });
      flywheelActive = false;
      flywheelBtn.classList.remove('on');
      flywheelBtn.textContent = 'FLYWHEEL';
    }, 500);
  }

  // --- Track & Shoot toggle ---
  const trackShootBtn = document.getElementById('track-shoot-btn');
  let trackShootActive = false;
  function toggleTrackShoot() {
    trackShootActive = !trackShootActive;
    trackShootBtn.classList.toggle('on', trackShootActive);
    trackShootBtn.textContent = trackShootActive ? 'TRACK&SHOOT ON' : 'TRACK&SHOOT';
    fetch('/track_shoot', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({state: trackShootActive})
    });
    if (!trackShootActive) {
      // UI sync: sweep off, flywheel/trigger off
      sweepActive = false;
      sweepBtn.classList.remove('on');
      sweepBtn.textContent = 'SWEEP';
      flywheelActive = false;
      flywheelBtn.classList.remove('on');
      flywheelBtn.textContent = 'FLYWHEEL';
      triggerActive = false;
      triggerBtn.classList.remove('on');
    }
  }

  // --- Joystick control (servo 0 = L/R, servo 2 = U/D) ---
  const joystick = document.getElementById('joystick');
  const knob = document.getElementById('joystick-knob');
  let joyActive = false;
  let joyRect = null;
  // Target angles — server does all smoothing
  let targetX = 88, targetY = 67;
  let lastSentX = 88, lastSentY = 67;

  // Send target to server periodically (no client-side interpolation)
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
    // Clamp to square
    dx = Math.max(-r, Math.min(r, dx));
    dy = Math.max(-r, Math.min(r, dy));
    // Position knob (follows finger/mouse)
    knob.style.left = (r + dx) + 'px';
    knob.style.top = (r + dy) + 'px';
    // Map to target servo angles (invert dy: drag up = negative dy = tilt up = lower angle)
    targetX = Math.round(90 - (dx / r) * 90);
    targetY = Math.round(65 + (dy / r) * 35);
    targetX = Math.max(0, Math.min(180, targetX));
    targetY = Math.max(30, Math.min(100, targetY));
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

  // --- Sync joystick knob with actual servo position when user isn't touching it ---
  setInterval(() => {
    if (joyActive) return;
    fetch('/status')
      .then(r => r.json())
      .then(data => {
        if (joyActive) return;
        const angles = data.servo_angles;
        if (!angles) return;
        const ax = angles['0'] ?? 90;
        const ay = angles['2'] ?? 100;
        targetX = ax; targetY = ay;
        lastSentX = ax; lastSentY = ay;
        // Convert angles back to knob pixel position
        const r = joystick.offsetWidth / 2;
        const dx = (90 - ax) / 90 * r;
        const dy = (ay - 65) / 35 * r;
        knob.style.left = (r + dx) + 'px';
        knob.style.top = (r + dy) + 'px';
      })
      .catch(() => {});
  }, 200);

  // --- Tab switching ---
  function switchTab(name) {
    document.querySelectorAll('.tab-btn').forEach((b, i) => {
      b.classList.toggle('active', (i === 0 && name === 'control') || (i === 1 && name === 'calibration'));
    });
    document.getElementById('tab-control').classList.toggle('active', name === 'control');
    document.getElementById('tab-calibration').classList.toggle('active', name === 'calibration');
    if (name === 'calibration') startCalFeed();
    else stopCalFeed();
  }

  // --- Calibration ---
  const calFeed = document.getElementById('cal-feed');
  const calOverlay = document.getElementById('cal-overlay');
  const calCoords = document.getElementById('cal-coords');
  const calSubmit = document.getElementById('cal-submit');
  let calInterval = null;
  let calPendingX = null, calPendingY = null;  // normalized 0-1
  let calSavedX = 0.5, calSavedY = 0.5;

  // Load saved calibration on startup
  fetch('/calibration').then(r => r.json()).then(data => {
    calSavedX = data.x; calSavedY = data.y;
    calCoords.textContent = 'Saved: (' + (data.x * 100).toFixed(1) + '%, ' + (data.y * 100).toFixed(1) + '%)';
  }).catch(() => {});

  function startCalFeed() {
    if (calInterval) return;
    fetchCalFrame();
    calInterval = setInterval(fetchCalFrame, 200);
    drawCalMarkers();
  }
  function stopCalFeed() {
    if (calInterval) { clearInterval(calInterval); calInterval = null; }
  }
  function fetchCalFrame() {
    fetch('/frame?' + Date.now())
      .then(r => r.blob())
      .then(blob => {
        const url = URL.createObjectURL(blob);
        calFeed.onload = () => { URL.revokeObjectURL(url); drawCalMarkers(); };
        calFeed.src = url;
      }).catch(() => {});
  }

  function drawCalMarkers() {
    const img = calFeed;
    if (!img.naturalWidth) return;
    const container = img.parentElement;
    const contW = container.clientWidth, contH = container.clientHeight;
    const imgW = img.naturalWidth, imgH = img.naturalHeight;
    // Compute displayed image rect (object-fit:contain)
    const scale = Math.min(contW / imgW, contH / imgH);
    const dispW = imgW * scale, dispH = imgH * scale;
    const offX = (contW - dispW) / 2, offY = (contH - dispH) / 2;

    let svg = '';
    // Draw saved calibration as a small circle
    const sx = offX + calSavedX * dispW, sy = offY + calSavedY * dispH;
    svg += '<circle cx="' + sx + '" cy="' + sy + '" r="6" fill="none" stroke="#00ff88" stroke-width="2"/>';
    svg += '<line x1="' + (sx-10) + '" y1="' + sy + '" x2="' + (sx+10) + '" y2="' + sy + '" stroke="#00ff88" stroke-width="1"/>';
    svg += '<line x1="' + sx + '" y1="' + (sy-10) + '" x2="' + sx + '" y2="' + (sy+10) + '" stroke="#00ff88" stroke-width="1"/>';

    // Draw pending mark as red X
    if (calPendingX !== null) {
      const px = offX + calPendingX * dispW, py = offY + calPendingY * dispH;
      const s = 14;
      svg += '<line x1="' + (px-s) + '" y1="' + (py-s) + '" x2="' + (px+s) + '" y2="' + (py+s) + '" stroke="#ff0000" stroke-width="3"/>';
      svg += '<line x1="' + (px+s) + '" y1="' + (py-s) + '" x2="' + (px-s) + '" y2="' + (py+s) + '" stroke="#ff0000" stroke-width="3"/>';
    }
    calOverlay.innerHTML = svg;
  }

  // Tap/click handler on calibration feed container
  document.querySelector('#tab-calibration .feed-container').addEventListener('click', function(e) {
    const img = calFeed;
    if (!img.naturalWidth) return;
    const container = this;
    const rect = container.getBoundingClientRect();
    const clickX = e.clientX - rect.left, clickY = e.clientY - rect.top;
    const contW = container.clientWidth, contH = container.clientHeight;
    const imgW = img.naturalWidth, imgH = img.naturalHeight;
    const scale = Math.min(contW / imgW, contH / imgH);
    const dispW = imgW * scale, dispH = imgH * scale;
    const offX = (contW - dispW) / 2, offY = (contH - dispH) / 2;
    // Check if click is within the image area
    const relX = clickX - offX, relY = clickY - offY;
    if (relX < 0 || relX > dispW || relY < 0 || relY > dispH) return;
    calPendingX = relX / dispW;
    calPendingY = relY / dispH;
    calSubmit.style.opacity = '1';
    calSubmit.style.pointerEvents = 'auto';
    calCoords.textContent = 'New: (' + (calPendingX * 100).toFixed(1) + '%, ' + (calPendingY * 100).toFixed(1) + '%)';
    drawCalMarkers();
  });
  // Touch support
  document.querySelector('#tab-calibration .feed-container').addEventListener('touchend', function(e) {
    if (e.changedTouches.length === 0) return;
    const touch = e.changedTouches[0];
    const rect = this.getBoundingClientRect();
    const evt = {clientX: touch.clientX, clientY: touch.clientY};
    // Simulate click
    const clickEvt = new MouseEvent('click', {clientX: touch.clientX, clientY: touch.clientY});
    this.dispatchEvent(clickEvt);
    e.preventDefault();
  });

  function submitCalibration() {
    if (calPendingX === null) return;
    fetch('/calibration', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({x: calPendingX, y: calPendingY})
    }).then(r => r.json()).then(data => {
      calSavedX = data.x; calSavedY = data.y;
      calPendingX = null; calPendingY = null;
      calSubmit.style.opacity = '0.4';
      calSubmit.style.pointerEvents = 'none';
      calCoords.textContent = 'Saved: (' + (data.x * 100).toFixed(1) + '%, ' + (data.y * 100).toFixed(1) + '%)';
      drawCalMarkers();
    }).catch(() => { calCoords.textContent = 'Save failed!'; });
  }

  function clearCalibration() {
    calPendingX = null; calPendingY = null;
    calSubmit.style.opacity = '0.4';
    calSubmit.style.pointerEvents = 'none';
    calCoords.textContent = 'Saved: (' + (calSavedX * 100).toFixed(1) + '%, ' + (calSavedY * 100).toFixed(1) + '%)';
    drawCalMarkers();
  }

  // --- Tap-to-aim on the control feed ---
  const feedContainer = document.querySelector('#tab-control .feed-container');
  function handleFeedTap(e) {
    const img = feed;
    if (!img.naturalWidth) return;
    const container = img.parentElement;
    const contRect = container.getBoundingClientRect();
    const imgW = img.naturalWidth, imgH = img.naturalHeight;
    const scale = Math.min(contRect.width / imgW, contRect.height / imgH);
    const dispW = imgW * scale, dispH = imgH * scale;
    const offX = (contRect.width - dispW) / 2;
    const offY = (contRect.height - dispH) / 2;
    const relX = (e.clientX - contRect.left - offX) / dispW;
    const relY = (e.clientY - contRect.top - offY) / dispH;
    if (relX < 0 || relX > 1 || relY < 0 || relY > 1) return;
    // Offset from calibrated aim center (positive = tap is right/below)
    const dx = relX - calSavedX;
    const dy = relY - calSavedY;
    // Convert to servo deltas, scaled by zoom (narrower FOV = smaller adjustment)
    const z = parseFloat(slider.value) || 1;
    const panDelta = -dx * 70 / z;
    const tiltDelta = dy * 40 / z;
    // Set new targets (overrides any previous movement)
    targetX = Math.max(0, Math.min(180, targetX + panDelta));
    targetY = Math.max(30, Math.min(100, targetY + tiltDelta));
    lastSentX = -1; lastSentY = -1;
    fetch('/servo', {method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({channel:0, angle:targetX})});
    fetch('/servo', {method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({channel:2, angle:targetY})});
  }
  feedContainer.addEventListener('click', handleFeedTap);
  feedContainer.addEventListener('touchend', function(e) {
    if (e.changedTouches.length === 0) return;
    const t = e.changedTouches[0];
    handleFeedTap({clientX: t.clientX, clientY: t.clientY});
    e.preventDefault();
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
servo_thread = threading.Thread(target=servo_smooth_worker, daemon=True)
servo_thread.start()
sweep_thread = threading.Thread(target=sweep_worker, daemon=True)
sweep_thread.start()
track_shoot_thread = threading.Thread(target=track_shoot_worker, daemon=True)
track_shoot_thread.start()
stream_thread = threading.Thread(target=stream_encoder, daemon=True)
stream_thread.start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=927, threaded=True)
