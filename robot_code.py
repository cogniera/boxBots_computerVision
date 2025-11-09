# # person_retreat_udp.py  (Python 3.11)
# # Camera: set URL to your stream (e.g., Iriun/OBS/ESP32 MJPEG) or use 0 for a USB webcam.

# import cv2, json, socket, time, math
# import numpy as np
# from ultralytics import YOLO

# # ====== CONFIG ======
# URL = 0  # replace with stream URL string if needed
# ESP_HOST = ""  # <- put your ESP32 IP here
# ESP_PORT = 
# SEND_HZ = 15
# CONF_THR = 0.5
# V_BACK = -0.25          # m/s-equivalent for your logic (sign only matters here)
# OMEGA_GAIN = 1.0        # turn strength based on horizontal offset
# SHOW_WINDOW = True
# # ====================

# sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# esp_addr = (ESP_HOST, ESP_PORT)

# def send_cmd(v, omega):
#     payload = {"cmd": "drive", "v": float(v), "omega": float(omega)}
#     sock.sendto(json.dumps(payload).encode("utf-8"), esp_addr)

# def send_stop():
#     payload = {"cmd": "stop"}
#     sock.sendto(json.dumps(payload).encode("utf-8"), esp_addr)

# # Load tiny YOLOv8n for person detection
# model = YOLO("yolov8n.pt")  # auto-download on first run

# cap = cv2.VideoCapture(URL)
# if not cap.isOpened():
#     raise RuntimeError("Could not open camera/stream; check URL or device index.")

# dt = 1.0 / SEND_HZ
# try:
#     last_send = 0.0
#     while True:
#         ok, frame = cap.read()
#         if not ok:
#             time.sleep(0.02)
#             continue

#         # Run detection (only ‘person’ class id 0 considered)
#         res = model.predict(source=frame, imgsz=480, conf=CONF_THR, verbose=False)[0]

#         H, W = frame.shape[:2]
#         person_found = False
#         best = None
#         for b in res.boxes:
#             cls_id = int(b.cls[0].item())
#             if cls_id != 0:
#                 continue
#             conf = float(b.conf[0].item())
#             x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
#             cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
#             wbox = (x2 - x1)
#             # pick the largest (closest-ish) person
#             if (best is None) or (wbox > best["w"]):
#                 best = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "cx": cx, "cy": cy, "w": wbox, "conf": conf}
#                 person_found = True

#         # Decide command
#         v_cmd, omega_cmd = 0.0, 0.0
#         if person_found:
#             # horizontal offset in [-1, 1], right = +, left = -
#             offset = (best["cx"] - (W / 2)) / (W / 2)
#             # retreat and turn AWAY from person (turn left if person on right, and vice-versa)
#             v_cmd = V_BACK
#             omega_cmd = OMEGA_GAIN * (-offset)  # negative offset -> turn right; positive -> turn left

#         # Send at fixed rate
#         now = time.monotonic()
#         if now - last_send >= dt:
#             if person_found:
#                 send_cmd(v_cmd, omega_cmd)
#             else:
#                 send_stop()
#             last_send = now

#         # Optional visualization
#         if SHOW_WINDOW:
#             if person_found:
#                 x1, y1, x2, y2 = int(best["x1"]), int(best["y1"]), int(best["x2"]), int(best["y2"])
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, f"RETREAT v={v_cmd:.2f} w={omega_cmd:.2f}", (10, 30),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
#             else:
#                 cv2.putText(frame, "NO PERSON -> STOP", (10, 30),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
#             cv2.imshow("Person Retreat Controller", frame)
#             if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
#                 break

# finally:
#     send_stop()
#     cap.release()
#     cv2.destroyAllWindows()


# python 3.11
# pip install opencv-python

# version 2: 

# import cv2, socket, json, time

# ESP_IP = ("", )   # <- change to your ESP32 IP
# URL = 1                         # 0 for webcam; or "http://<ip>:<port>/stream"

# sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# def send(obj): sock.sendto(json.dumps(obj).encode(), ESP_IP)

# # Haar cascade file usually ships with OpenCV; if not, download or point to its path
# face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# cap = cv2.VideoCapture(URL)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

# last_send = 0
# SEND_PERIOD = 1/15  # ~15 Hz

# while True:
#     ok, frame = cap.read()
#     if not ok:
#         time.sleep(0.05)
#         continue

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face.detectMultiScale(gray, 1.2, 5)

#     now = time.time()
#     if now - last_send >= SEND_PERIOD:
#         if len(faces) > 0:
#             # person seen -> retreat
#             send({"cmd":"drive","v":-0.2,"omega":0.0})
#         else:
#             # no person -> stop
#             send({"cmd":"stop"})
#         last_send = now

#     # (Optional preview)
#     for (x,y,w,h) in faces:
#         cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
#     cv2.imshow("Person detector", frame)
#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# Version 3: 

# import cv2, json, socket, time
# import numpy as np
# from ultralytics import YOLO

# # ========= CONFIG =========
# SOURCE = 1               # 1 = second webcam; use 0 for default, or "http://.../stream" for URL
# ESP_HOST = ""  # <-- put your ESP32 IP here
# ESP_PORT = 
# SEND_HZ = 15
# CONF_THR = 0.5
# IMG_SIZE = 480

# # Motion policy
# V_BACK = -0.25          # retreat speed "logical" value (ESP interprets it)
# OMEGA_GAIN = 1.0        # turn strength away from detected person (set 0.0 to disable)
# SHOW_WINDOW = True
# # =========================

# sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# esp_addr = (ESP_HOST, ESP_PORT)
# dt = 1.0 / SEND_HZ

# def send_cmd(v, omega):
#     payload = {"cmd": "drive", "v": float(v), "omega": float(omega)}
#     sock.sendto(json.dumps(payload).encode("utf-8"), esp_addr)

# def send_stop():
#     sock.sendto(json.dumps({"cmd": "stop"}).encode("utf-8"), esp_addr)

# # Load tiny YOLOv8n (auto-downloads weights on first run)
# model = YOLO("yolov8n.pt")

# cap = cv2.VideoCapture(SOURCE)
# if not cap.isOpened():
#     raise RuntimeError(f"Could not open SOURCE={SOURCE}. If using a URL, set SOURCE='http://<ip>:<port>/stream'")

# last_send = 0.0
# try:
#     while True:
#         ok, frame = cap.read()
#         if not ok:
#             time.sleep(0.02)
#             continue

#         # Inference
#         # We pass the numpy frame directly; ultralytics handles BGR/RGB internally
#         results = model.predict(frame, imgsz=IMG_SIZE, conf=CONF_THR, verbose=False)
#         res = results[0]

#         H, W = frame.shape[:2]
#         person_found = False
#         best = None  # keep largest bbox (roughly the closest)
#         for b in res.boxes:
#             cls_id = int(b.cls[0].item())
#             if cls_id != 0:   # class 0 = 'person' in COCO
#                 continue
#             x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
#             conf = float(b.conf[0].item())
#             wbox = (x2 - x1)
#             if (best is None) or (wbox > best["w"]):
#                 best = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "w": wbox, "conf": conf}
#                 person_found = True

#         # Decide v, omega
#         v_cmd, omega_cmd = 0.0, 0.0
#         if person_found:
#             cx = (best["x1"] + best["x2"]) * 0.5
#             offset = (cx - (W / 2)) / (W / 2)  # -1 (left edge) .. +1 (right edge)
#             v_cmd = V_BACK
#             omega_cmd = OMEGA_GAIN * (-offset)  # turn away from the person

#         # Send at fixed rate
#         now = time.monotonic()
#         if now - last_send >= dt:
#             if person_found:
#                 send_cmd(v_cmd, omega_cmd)
#             else:
#                 send_stop()
#             last_send = now

#         # Optional viz
#         if SHOW_WINDOW:
#             if person_found:
#                 x1, y1, x2, y2 = map(int, (best["x1"], best["y1"], best["x2"], best["y2"]))
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, f"RETREAT v={v_cmd:.2f} w={omega_cmd:.2f}",
#                             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
#             else:
#                 cv2.putText(frame, "NO PERSON -> STOP",
#                             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
#             cv2.imshow("YOLOv8n Person Retreat", frame)
#             if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
#                 break
# finally:
#     send_stop()
#     cap.release()
#     cv2.destroyAllWindows()

# version 4 : 

# import cv2, socket, json, time
# import numpy as np
# import torch
# import torch.nn.functional as F
# from ultralytics import YOLO

# # ========= USER CONFIG =========
# SOURCE = 1                     # webcam index 1
# ESP_HOST = ""      # <-- change to your ESP32 IP
# ESP_PORT = 
# SEND_HZ = 15

# CONF_THR = 0.4                 # YOLO person confidence threshold
# IMG_SIZE = 480                 # YOLO input size
# V_BACK = -0.25                 # retreat speed
# OMEGA_GAIN = 1.0               # turn strength away from person
# PERSON_NEAR_THR = 0.55         # depth closeness threshold [0..1]
# SHOW = True                    # display window
# # ===============================

# # UDP socket
# sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# ESP_ADDR = (ESP_HOST, ESP_PORT)

# def send_cmd(v, omega):
#     sock.sendto(json.dumps(
#         {"cmd":"drive","v":float(v),"omega":float(omega)}
#     ).encode("utf-8"), ESP_ADDR)

# def send_stop():
#     sock.sendto(json.dumps({"cmd":"stop"}).encode("utf-8"), ESP_ADDR)

# # Load YOLO
# yolo = YOLO("yolov8n.pt")

# # Load MiDaS depth model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).to(device).eval()
# midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
# transform = midas_transforms.small_transform  # <- use with MiDaS_small

# def depth_map(frame):
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     inp = transform(rgb).to(device)
#     with torch.no_grad():
#         pred = midas(inp)
#         pred = F.interpolate(pred.unsqueeze(1), size=rgb.shape[:2],
#                              mode="bicubic", align_corners=False).squeeze(1)
#         d = pred[0].cpu().numpy()
#     # Normalize depth to [0,1]
#     d_min, d_max = np.percentile(d, 2), np.percentile(d, 98)
#     return np.clip((d - d_min) / (d_max - d_min + 1e-6), 0, 1)

# # Open video
# cap = cv2.VideoCapture(SOURCE)
# if not cap.isOpened():
#     raise RuntimeError("Could not open camera SOURCE=1")

# dt = 1.0 / SEND_HZ
# last_send = 0.0

# try:
#     while True:
#         ok, frame = cap.read()
#         if not ok:
#             time.sleep(0.02)
#             continue

#         H, W = frame.shape[:2]

#         # YOLO detect people
#         res = yolo.predict(frame, imgsz=IMG_SIZE, conf=CONF_THR, verbose=False)[0]
#         best = None
#         for b in res.boxes:
#             if int(b.cls[0].item()) != 0:  # class 0 = person
#                 continue
#             x1,y1,x2,y2 = b.xyxy[0].cpu().numpy().tolist()
#             wbox = x2 - x1
#             if best is None or wbox > best["w"]:
#                 best = {"x1":x1,"y1":y1,"x2":x2,"y2":y2,"w":wbox}

#         # Compute depth
#         dn = depth_map(frame)

#         v_cmd, omega_cmd = 0.0, 0.0
#         person_close = False
#         median_d = None  # for display safety

#         if best is not None:
#             x1,y1,x2,y2 = map(int,(best["x1"],best["y1"],best["x2"],best["y2"]))
#             x1=max(0,min(W-1,x1)); x2=max(0,min(W-1,x2))
#             y1=max(0,min(H-1,y1)); y2=max(0,min(H-1,y2))
#             box = dn[y1:y2, x1:x2]
#             if box.size > 0:
#                 median_d = float(np.median(box))

#                 if median_d >= PERSON_NEAR_THR:
#                     person_close = True
#                     cx = 0.5*(x1+x2)
#                     offset = (cx-(W/2))/(W/2)   # -1 left, +1 right
#                     v_cmd = V_BACK
#                     omega_cmd = -offset * OMEGA_GAIN

#         # --- send at fixed rate + console log ---
#         now = time.monotonic()
#         if now - last_send >= dt:
#             if person_close:
#                 send_cmd(v_cmd, omega_cmd)
#                 print(f"[SEND] RETREAT v={v_cmd:.3f}, omega={omega_cmd:.3f}")
#             else:
#                 send_stop()
#                 print("[SEND] STOP")
#             last_send = now

#         if SHOW:
#             disp = frame.copy()

#             # Draw person box + median depth (if any)
#             if best is not None:
#                 cv2.rectangle(disp,(x1,y1),(x2,y2),(0,255,0),2)
#                 if median_d is not None:
#                     cv2.putText(disp,f"median={median_d:.2f}",(x1,max(20,y1-10)),
#                                 cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

#             # --- HUD overlay with current command (THIS IS WHAT YOU ASKED FOR) ---
#             if person_close:
#                 hud_text = f"RETREAT  v={v_cmd:.2f}  \u03C9={omega_cmd:.2f}"
#                 hud_color = (0, 0, 255)  # red
#             else:
#                 hud_text = "STOP"
#                 hud_color = (0, 255, 255)  # yellow

#             # semi-transparent bar behind HUD text for readability
#             bar_h = 36
#             overlay = disp.copy()
#             cv2.rectangle(overlay, (0,0), (W, bar_h+10), (0,0,0), -1)
#             disp = cv2.addWeighted(overlay, 0.35, disp, 0.65, 0)

#             cv2.putText(disp, hud_text, (10, bar_h),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, hud_color, 2, cv2.LINE_AA)

#             # Small status under the HUD if a person is detected but far
#             if (best is not None) and not person_close and (median_d is not None):
#                 cv2.putText(disp, "PERSON FAR", (10, bar_h+28),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)

#             # Depth mini-view
#             dn_small = cv2.resize(dn,(W//4,H//4))
#             dn_vis = cv2.applyColorMap((dn_small*255).astype(np.uint8), cv2.COLORMAP_INFERNO)
#             disp[0:dn_vis.shape[0], 0:dn_vis.shape[1]] = dn_vis

#             cv2.imshow("YOLO + MiDaS Retreat", disp)
#             if cv2.waitKey(1)&0xFF==27:
#                 break

# finally:
#     send_stop()
#     cap.release()
#     cv2.destroyAllWindows()

#version 5 

import cv2, socket, json, time, threading, queue, math
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

# ========= USER CONFIG =========
SOURCE = 1                        # your webcam index (use 0/1/2...) or a URL string
ESP_HOST = ""          # ESP32 IP
ESP_PORT =                    # ESP32 UDP command port
PC_LISTEN_PORT =              # PC UDP port to RECEIVE ultrasonic telemetry from ESP32
SEND_HZ = 15

CONF_THR = 0.45                   # YOLO person confidence
IMG_SIZE = 480                    # YOLO inference size
V_BACK = -0.25                    # retreat speed (abstract m/s)
OMEGA_GAIN = 1.15                 # turn strength away from person center
PERSON_NEAR_THR = 0.55            # depth closeness threshold [0..1] (MiDaS normalized: higher = closer after our normalization)
EMA_ALPHA = 0.35                  # smoothing for median depth

# Wall safety (from ESP32 ultrasonic)
R_CRIT_CM = 20                    # hard-stop distance
SLIDE_TIME = 0.60                 # seconds to slide along wall
BOUNCE_TIME = 0.50                # seconds to “bounce” away (turn + slight forward)
SLIDE_SPEED = -0.12               # retreat speed while sliding
BOUNCE_SPEED = 0.12               # forward speed while bouncing
SAFE_OMEGA = 0.9                  # turn rate used for slide/bounce
# ===============================

SHOW = True

# ---------- UDP: send to ESP32 ----------
cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ESP_ADDR = (ESP_HOST, ESP_PORT)

def send_cmd(v, omega):
    payload = {"cmd": "drive", "v": float(v), "omega": float(omega)}
    cmd_sock.sendto(json.dumps(payload).encode("utf-8"), ESP_ADDR)

def send_stop():
    payload = {"cmd": "stop"}
    cmd_sock.sendto(json.dumps(payload).encode("utf-8"), ESP_ADDR)

# ---------- UDP: receive telemetry from ESP32 (range_cm) ----------
telemetry_q = queue.Queue()

def telemetry_listener():
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("0.0.0.0", PC_LISTEN_PORT))
    rx.settimeout(0.5)
    while True:
        try:
            data, _ = rx.recvfrom(1024)
            msg = json.loads(data.decode("utf-8"))
            telemetry_q.put(msg)
        except socket.timeout:
            pass
        except Exception:
            pass

threading.Thread(target=telemetry_listener, daemon=True).start()

# ---------- Camera (use DirectShow on Windows to avoid MSMF grab errors) ----------
# If SOURCE is int, prefer DirectShow; if it’s a URL string, standard open.
if isinstance(SOURCE, int):
    cap = cv2.VideoCapture(SOURCE, cv2.CAP_DSHOW)
else:
    cap = cv2.VideoCapture(SOURCE)

if not cap.isOpened():
    raise RuntimeError(f"Could not open camera/stream: {SOURCE}")

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ---------- Models ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# YOLO person detector
yolo = YOLO("yolov8n.pt")

# MiDaS small + proper transform
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).to(device).eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
transform = midas_transforms.small_transform  # use with MiDaS_small

# optional speedups
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

def depth_map(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    inp = transform(rgb).to(device)
    with torch.no_grad():
        pred = midas(inp)
        pred = F.interpolate(
            pred.unsqueeze(1), size=rgb.shape[:2], mode="bicubic", align_corners=False
        ).squeeze(1)
        d = pred[0].detach().cpu().numpy()
    # Normalize depth to [0,1] (invert-ish by percentile so "near" ~ bigger value after normalization we choose)
    d_min, d_max = np.percentile(d, 2), np.percentile(d, 98)
    dn = np.clip((d - d_min) / (d_max - d_min + 1e-6), 0, 1)
    return dn

# ---------- Control loop ----------
dt = 1.0 / SEND_HZ
last_send = 0.0
ema_depth = None
ultra_cm = None
mode = "IDLE"     # IDLE, RETREAT, SLIDE, BOUNCE
phase_t0 = 0.0
last_offset = 0.0 # remember where the person was (left/right) for slide/bounce orientation

try:
    while True:
        # ingest latest telemetry (non-blocking)
        while not telemetry_q.empty():
            msg = telemetry_q.get_nowait()
            if "range_cm" in msg:
                try:
                    ultra_cm = float(msg["range_cm"])
                except:
                    pass

        ok, frame = cap.read()
        if not ok:
            time.sleep(0.02)
            continue

        H, W = frame.shape[:2]

        # --- YOLO detect person (take the largest bbox) ---
        res = yolo.predict(frame, imgsz=IMG_SIZE, conf=CONF_THR, verbose=False)[0]
        best = None
        for b in res.boxes:
            if int(b.cls[0].item()) != 0:  # person only
                continue
            x1, y1, x2, y2 = b.xyxy[0].detach().cpu().numpy().tolist()
            wbox = x2 - x1
            if (best is None) or (wbox > best["w"]):
                best = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "w": wbox}

        # --- Depth ---
        dn = depth_map(frame)

        # --- Decide high-level command ---
        v_cmd, omega_cmd = 0.0, 0.0
        person_close = False
        median_d = None

        if best is not None:
            x1, y1, x2, y2 = map(int, (best["x1"], best["y1"], best["x2"], best["y2"]))
            x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W - 1, x2))
            y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H - 1, y2))
            roi = dn[y1:y2, x1:x2]
            if roi.size > 0:
                median_d = float(np.median(roi))
                # smooth the depth
                ema_depth = median_d if ema_depth is None else (EMA_ALPHA * median_d + (1-EMA_ALPHA) * ema_depth)
                cx = 0.5 * (x1 + x2)
                offset = (cx - (W/2)) / (W/2)  # -1 left, +1 right
                last_offset = offset  # remember direction

                # Person is "near" only if normalized depth passes threshold
                if ema_depth is not None and ema_depth >= PERSON_NEAR_THR:
                    person_close = True

        now = time.monotonic()

        # --- Mode machine ---
        if mode in ("IDLE", "RETREAT"):
            if person_close:
                mode = "RETREAT"
                v_cmd = V_BACK
                omega_cmd = -last_offset * OMEGA_GAIN   # turn away from person
                # wall safety: if we see a wall very near while retreating, go to SLIDE
                if ultra_cm is not None and ultra_cm <= R_CRIT_CM:
                    mode = "SLIDE"
                    phase_t0 = now
            else:
                mode = "IDLE"
                v_cmd, omega_cmd = 0.0, 0.0

        if mode == "SLIDE":
            # slide parallel to wall (turn in place a bit toward parallel, then retreat slowly)
            elapsed = now - phase_t0
            # choose sign: if person was on the right (offset>0), we turned left while retreating,
            # so to "hug/slide", we keep turning slightly the SAME direction to get parallel.
            sign = -1.0 if last_offset > 0 else 1.0
            omega_cmd = sign * SAFE_OMEGA
            v_cmd = SLIDE_SPEED
            if elapsed >= SLIDE_TIME:
                mode = "BOUNCE"
                phase_t0 = now

        if mode == "BOUNCE":
            # push away from wall a bit (turn opposite sign and go forward slightly)
            elapsed = now - phase_t0
            sign = 1.0 if last_offset > 0 else -1.0  # opposite of slide
            omega_cmd = sign * SAFE_OMEGA
            v_cmd = BOUNCE_SPEED
            if elapsed >= BOUNCE_TIME:
                # after bounce, return to IDLE/RETREAT depending on current person_close
                mode = "RETREAT" if person_close else "IDLE"

        # --- Send at fixed rate ---
        if now - last_send >= dt:
            if v_cmd != 0.0 or omega_cmd != 0.0:
                send_cmd(v_cmd, omega_cmd)
                print(f"[{mode}] v={v_cmd:.2f}  w={omega_cmd:.2f}  depth_med={ema_depth:.2f if ema_depth else -1:.2f}  range={ultra_cm}")
            else:
                send_stop()
                print(f"[IDLE] stop  range={ultra_cm}")
            last_send = now

        # --- Viz ---
        if SHOW:
            disp = frame.copy()
            # person box
            if best is not None:
                x1, y1, x2, y2 = map(int, (best["x1"], best["y1"], best["x2"], best["y2"]))
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0,255,0), 2)
                if ema_depth is not None:
                    cv2.putText(disp, f"depth_med(EMA)={ema_depth:.2f}", (x1, max(20, y1-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            # HUD
            Wd, Hd = disp.shape[1], disp.shape[0]
            bar_h = 38
            overlay = disp.copy()
            cv2.rectangle(overlay, (0,0), (Wd, bar_h+12), (0,0,0), -1)
            disp = cv2.addWeighted(overlay, 0.35, disp, 0.65, 0)

            if mode == "IDLE":
                hud_text = "STOP"
                color = (0,255,255)
            elif mode == "RETREAT":
                hud_text = "RETREAT"
                color = (0,0,255)
            elif mode == "SLIDE":
                hud_text = "SLIDE (wall)"
                color = (255,0,0)
            else:
                hud_text = "BOUNCE"
                color = (255,0,255)

            cv2.putText(disp, f"{hud_text} | range={ultra_cm}cm", (10, bar_h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

            # depth mini-view
            dn_small = cv2.resize(dn, (Wd//4, Hd//4))
            dn_vis = cv2.applyColorMap((dn_small*255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            disp[0:dn_vis.shape[0], 0:dn_vis.shape[1]] = dn_vis

            cv2.imshow("YOLO + MiDaS Retreat (v2)", disp)
            if cv2.waitKey(1) & 0xFF == 27:
                break

finally:
    send_stop()
    cap.release()

    cv2.destroyAllWindows()
