"""
VIT Elite Kiosk Attendance — Professional UI
═════════════════════════════════════════════
InsightFace ArcFace  |  Same vault as enrollment.py
Translucent UI  |  Scan animation  |  Time slots  |  CSV log
"""

import cv2
import pickle
import pandas as pd
import numpy as np
import os
import math
import time
from insightface.app import FaceAnalysis
from ultralytics import YOLO
from datetime import datetime
import torch

# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════
VAULT_PATH  = r"data\biometric_vault.pkl"
LIVE_CSV    = r"data\live_attendance.csv"
THRESHOLD   = 0.20
MIN_FACE_PX = 20
FRAME_SKIP  = 3

# ══════════════════════════════════════════════════════════════
#  COLOURS (BGR)
# ══════════════════════════════════════════════════════════════
C_WHITE  = (255, 255, 255)
C_BLACK  = ( 10,  10,  10)
C_GREEN  = ( 34, 197,  94)
C_RED    = ( 59,  68, 239)
C_AMBER  = (  0, 165, 255)
C_BLUE   = (255, 140,   0)
C_YELLOW = (  0, 220, 255)
C_DARK   = (  8,   8,  12)
C_GRAY   = (130, 130, 130)
FONT     = cv2.FONT_HERSHEY_DUPLEX
AA       = cv2.LINE_AA

# ══════════════════════════════════════════════════════════════
#  TIME SLOTS
# ══════════════════════════════════════════════════════════════
def get_slot_info():
    now = datetime.now()
    schedule = [
        ("Theory 1", "08:00", "08:50"),
        ("Theory 2", "09:00", "09:50"),
        ("Theory 3", "10:00", "10:50"),
        ("Theory 4", "11:00", "11:50"),
        ("Lab 1",    "14:00", "15:40"),
        ("Lab 2",    "15:51", "17:30"),
        ("Lab 3",    "17:40", "19:20")
    ]
    for slot, start, end in schedule:
        s = datetime.strptime(start, "%H:%M").time()
        e = datetime.strptime(end,   "%H:%M").time()
        if s <= now.time() <= e:
            diff = (datetime.combine(now.date(), now.time()) -
                    datetime.combine(now.date(), s)).total_seconds() / 60
            return slot, ("ALLOWED" if diff <= 15 else "LATE")
    return "Campus Access", "PAUSED"

# ══════════════════════════════════════════════════════════════
#  GPU + MODELS
# ══════════════════════════════════════════════════════════════
print("="*60)
print("  VIT ELITE KIOSK  —  ArcFace + Professional UI")
print("="*60)

if torch.cuda.is_available():
    YOLO_DEVICE  = 0
    IF_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    GPU_LABEL    = "GPU: CUDA"
    print(f"[GPU] {torch.cuda.get_device_name(0)}")
else:
    YOLO_DEVICE  = "cpu"
    IF_PROVIDERS = ["CPUExecutionProvider"]
    GPU_LABEL    = "CPU"
    print("[CPU] mode")

yolo = YOLO(r"models\yolov8n-face.pt")
dummy = np.zeros((640, 640, 3), dtype=np.uint8)
yolo.predict(dummy, verbose=False, device=YOLO_DEVICE)
print("[YOLOv8] Ready.")

face_app = FaceAnalysis(name="buffalo_l", providers=IF_PROVIDERS)
face_app.prepare(ctx_id=0, det_size=(320, 320))
rec_model = face_app.models.get("recognition")
print("[ArcFace] Ready.")

if not os.path.exists(VAULT_PATH):
    print("[ERROR] Vault not found. Run enrollment.py first.")
    exit(1)
with open(VAULT_PATH, "rb") as f:
    vault = pickle.load(f)
print(f"[Vault] {len(vault)} student(s) loaded.")

os.makedirs("data", exist_ok=True)
already_marked = set()

# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════
def emb_list(s):
    return s if isinstance(s, list) else [s]

def cosine(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / (d + 1e-9))

def get_embedding(crop_bgr):
    try:
        if crop_bgr is None: return None
        h, w = crop_bgr.shape[:2]
        if w < MIN_FACE_PX or h < MIN_FACE_PX: return None
        if w < 112 or h < 112:
            scale = max(112.0/w, 112.0/h) * 1.5
            nw = max(int(w*scale), 112)
            nh = max(int(h*scale), 112)
            crop_bgr = cv2.resize(crop_bgr, (nw, nh),
                                  interpolation=cv2.INTER_LANCZOS4)
            if crop_bgr is None or crop_bgr.size == 0: return None
        rgb   = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        faces = face_app.get(rgb)
        if faces:
            best = max(faces,
                       key=lambda f:(f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            if best.embedding is not None:
                return best.embedding
        if rec_model:
            r112 = cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_LINEAR)
            inp  = np.expand_dims(r112.astype(np.float32), 0)
            return rec_model.get_feat(inp).flatten()
        return None
    except: return None

def match_vault(vec):
    best_s, best_n, best_r = 0.0, "Unknown", "N/A"
    for n, (stored, reg) in vault.items():
        s = max(cosine(vec, e) for e in emb_list(stored))
        if s > best_s:
            best_s, best_n, best_r = s, n, reg
    if best_s >= THRESHOLD:
        return best_n, best_r, best_s
    return "Unknown", "N/A", best_s

def log_to_csv(name, reg, slot, status):
    row = pd.DataFrame([[
        name, reg,
        datetime.now().strftime("%Y-%m-%d"),
        datetime.now().strftime("%H:%M:%S"),
        slot, status
    ]], columns=["Name","RegNo","Date","Time","Slot","Status"])
    row.to_csv(LIVE_CSV, mode="a",
               header=not os.path.exists(LIVE_CSV), index=False)

# ══════════════════════════════════════════════════════════════
#  DRAWING
# ══════════════════════════════════════════════════════════════
def draw_grid(frame):
    H, W = frame.shape[:2]
    ov = frame.copy()
    for x in range(0, W, 80):
        cv2.line(ov, (x, 62), (x, H-130), (255,255,255), 1, AA)
    for y in range(62, H-130, 60):
        cv2.line(ov, (0, y),  (W, y),     (255,255,255), 1, AA)
    cv2.addWeighted(ov, 0.04, frame, 0.96, 0, frame)

def draw_scan_line(frame, tick):
    H, W = frame.shape[:2]
    y = int(62 + abs(math.sin(tick * 0.03)) * (H - 192))
    cv2.line(frame, (0, y), (W, y), C_GREEN, 1, AA)
    ov = frame.copy()
    for dy in [1, 2, 3]:
        cv2.line(ov, (0,y+dy), (W,y+dy), C_GREEN, 1, AA)
    cv2.addWeighted(ov, 0.06, frame, 0.94, 0, frame)

def draw_corner_ui(frame):
    H, W = frame.shape[:2]
    L = 40; T = 3; col = (50, 60, 80)
    for px,py,dx,dy in [(0,62,1,1),(W,62,-1,1),
                        (0,H-130,1,-1),(W,H-130,-1,-1)]:
        cv2.line(frame,(px,py),(px+dx*L,py),col,T,AA)
        cv2.line(frame,(px,py),(px,py+dy*L),col,T,AA)

def draw_scan_ring(frame, cx, cy, rx, ry, angle, col):
    axes = (rx+18, ry+18)
    cv2.ellipse(frame,(cx,cy),axes, angle,    0, 270, col, 2, AA)
    cv2.ellipse(frame,(cx,cy),axes, angle+180, 0,  90, col, 2, AA)

def draw_face_box(frame, x1, y1, x2, y2, col):
    cv2.rectangle(frame,(x1,y1),(x2,y2),col,1,AA)
    L = max(12, min(24,(x2-x1)//4))
    for px,py,dx,dy in [(x1,y1,1,1),(x2,y1,-1,1),
                        (x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame,(px,py),(px+dx*L,py),col,3,AA)
        cv2.line(frame,(px,py),(px,py+dy*L),col,3,AA)

def draw_label(frame, x1, y1, text, col):
    fs = 0.58
    (tw,th),bl = cv2.getTextSize(text, FONT, fs, 1)
    pad = 8
    lx2 = x1 + tw + pad*2
    ly1 = max(0, y1 - th - bl - pad*2)
    ov  = frame.copy()
    cv2.rectangle(ov,(x1,ly1),(lx2,y1),col,cv2.FILLED)
    cv2.addWeighted(ov,0.75,frame,0.25,0,frame)
    cv2.putText(frame,text,(x1+pad,y1-bl-pad//2),FONT,fs,C_WHITE,1,AA)

def draw_conf_bar(frame, x1, y1, x2, score):
    bw     = x2 - x1
    filled = int(bw * min(score/0.6, 1.0))
    cv2.rectangle(frame,(x1,y1+4),(x2,y1+10),(40,40,40),cv2.FILLED)
    col = C_GREEN if score >= THRESHOLD else C_RED
    cv2.rectangle(frame,(x1,y1+4),(x1+filled,y1+10),col,cv2.FILLED)

def draw_top_bar(frame):
    H, W = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov,(0,0),(W,62),(8,8,12),cv2.FILLED)
    cv2.addWeighted(ov, 0.88, frame, 0.12, 0, frame)
    cv2.line(frame,(0,62),(W,62),(50,50,70),1,AA)

    cv2.putText(frame, "VIT",  (16,44), FONT, 1.10, C_YELLOW, 2, AA)
    cv2.putText(frame, "SMART ATTENDANCE SYSTEM",
                (90,44), FONT, 0.72, C_WHITE, 1, AA)

    clock = datetime.now().strftime("%I:%M:%S %p")
    (cw,_),_ = cv2.getTextSize(clock, FONT, 0.68, 1)
    cv2.putText(frame, clock, (W-cw-16,44), FONT, 0.68, C_WHITE, 1, AA)

    slot, stat = get_slot_info()
    sc = C_GREEN if stat=="ALLOWED" else (C_AMBER if stat=="LATE" else C_GRAY)
    slot_txt = f"{slot}  [{stat}]"
    (sw,_),_ = cv2.getTextSize(slot_txt, FONT, 0.55, 1)
    cv2.putText(frame, slot_txt, (W//2-sw//2,44), FONT, 0.55, sc, 1, AA)

    (gw,_),_ = cv2.getTextSize(GPU_LABEL, FONT, 0.48, 1)
    cv2.putText(frame, GPU_LABEL, (W-cw-gw-30,22), FONT, 0.48, C_GREEN, 1, AA)

def draw_bottom_panel(frame, status_text, status_col, student_info):
    H, W = frame.shape[:2]
    ph   = 130
    ov   = frame.copy()
    cv2.rectangle(ov,(0,H-ph),(W,H),(8,8,12),cv2.FILLED)
    cv2.addWeighted(ov, 0.88, frame, 0.12, 0, frame)
    cv2.line(frame,(0,H-ph),(W,H-ph),(50,50,70),1,AA)

    cv2.putText(frame,"STATUS",  (24,H-ph+28),FONT,0.45,C_GRAY,1,AA)
    cv2.putText(frame,status_text,(24,H-ph+65),FONT,0.80,status_col,2,AA)

    cv2.line(frame,(W//2,H-ph+10),(W//2,H-10),(50,50,70),1,AA)

    cv2.putText(frame,"BIOMETRIC TARGET",
                (W//2+20,H-ph+28),FONT,0.45,C_GRAY,1,AA)
    cv2.putText(frame,student_info,
                (W//2+20,H-ph+65),FONT,0.70,C_WHITE,1,AA)

    date_str = datetime.now().strftime("%d %B %Y")
    (dw,_),_ = cv2.getTextSize(date_str,FONT,0.45,1)
    cv2.putText(frame,date_str,(W-dw-16,H-10),FONT,0.45,C_GRAY,1,AA)
    cv2.putText(frame,f"VAULT: {len(vault)} enrolled",
                (16,H-10),FONT,0.45,C_GRAY,1,AA)

def draw_flash(frame, col, alpha=0.15):
    ov = frame.copy()
    cv2.rectangle(ov,(0,0),(frame.shape[1],frame.shape[0]),col,cv2.FILLED)
    cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)

# ══════════════════════════════════════════════════════════════
#  CAMERA
# ══════════════════════════════════════════════════════════════
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

cv2.namedWindow("VIT Kiosk", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("VIT Kiosk", cv2.WND_PROP_FULLSCREEN,
                      cv2.WINDOW_FULLSCREEN)

# ══════════════════════════════════════════════════════════════
#  STATE
# ══════════════════════════════════════════════════════════════
scan_angle   = 0
frame_no     = 0
flash_frames = 0
flash_col    = C_GREEN
hold_frames  = 0
last_status  = "SCANNING..."
last_col     = C_WHITE
last_detail  = "SEARCHING FOR BIOMETRIC TARGET"

# ══════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════
while True:
    ret, frame = cap.read()
    if not ret: break
    frame_no += 1
    H, W = frame.shape[:2]

    # Background effects
    draw_grid(frame)
    draw_scan_line(frame, frame_no)
    draw_corner_ui(frame)

    face_found = False

    if frame_no % FRAME_SKIP == 0:
        res = yolo.predict(frame, conf=0.60, verbose=False,
                           device=YOLO_DEVICE, imgsz=640)

        if res[0].boxes and len(res[0].boxes):
            for box in res[0].boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0].cpu().numpy())
                bw, bh = x2-x1, y2-y1
                if bw < MIN_FACE_PX or bh < MIN_FACE_PX: continue

                face_found = True
                cx = (x1+x2)//2; cy = (y1+y2)//2

                try:
                    crop = frame[max(0,y1):min(H,y2),
                                 max(0,x1):min(W,x2)]
                    vec  = get_embedding(crop)

                    if vec is not None:
                        name, reg, score = match_vault(vec)
                        slot, time_stat  = get_slot_info()

                        if name != "Unknown":
                            col = C_GREEN
                            if name in already_marked:
                                last_status = f"ACCESS ALREADY LOGGED — {name}"
                                last_col    = C_AMBER
                            else:
                                if time_stat == "ALLOWED":
                                    log_to_csv(name,reg,slot,"Present")
                                    already_marked.add(name)
                                    last_status = "ACCESS GRANTED — ATTENDANCE RECORDED"
                                    last_col    = C_GREEN
                                    flash_col   = C_GREEN
                                    flash_frames= 10
                                elif time_stat == "LATE":
                                    log_to_csv(name,reg,slot,"Late")
                                    already_marked.add(name)
                                    last_status = "ACCESS RESTRICTED — LATE ENTRY LOGGED"
                                    last_col    = C_AMBER
                                    flash_col   = C_AMBER
                                    flash_frames= 10
                                else:
                                    last_status = "ENTRY DENIED — CLASS NOT IN SESSION"
                                    last_col    = C_RED
                            last_detail = f"{name}  |  REG: {reg}  |  {slot}"
                            hold_frames = 50
                        else:
                            col = C_RED
                            last_status = "UNIDENTIFIED SUBJECT"
                            last_col    = C_RED
                            last_detail = "NOT FOUND IN BIO-VAULT"
                            hold_frames = 20

                        draw_face_box(frame,x1,y1,x2,y2,col)
                        draw_scan_ring(frame,cx,cy,bw//2,bh//2,
                                       scan_angle,col)
                        lbl = (f"{name}  {reg}  ({score:.2f})"
                               if name != "Unknown" else "Unknown")
                        draw_label(frame,x1,y1,x2,lbl,col)
                        draw_conf_bar(frame,x1,y2,x2,score)

                    else:
                        draw_face_box(frame,x1,y1,x2,y2,C_GRAY)

                except: pass

    if not face_found:
        if hold_frames > 0:
            hold_frames -= 1
        else:
            last_status = "SCANNING..."
            last_col    = C_WHITE
            last_detail = "SEARCHING FOR BIOMETRIC TARGET"

    if flash_frames > 0:
        draw_flash(frame, flash_col)
        flash_frames -= 1

    scan_angle = (scan_angle + 4) % 360

    # UI panels drawn LAST so they sit on top
    draw_top_bar(frame)
    draw_bottom_panel(frame, last_status, last_col, last_detail)

    cv2.imshow("VIT Kiosk", cv2.resize(frame, (1920, 1080)))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print(f"[Done] {len(already_marked)} students marked.")