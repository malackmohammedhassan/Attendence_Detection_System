"""
VIT Elite CCTV Attendance — process_video.py  (v5 + name-after-register fix)
Key fixes:
  1. Once a name is CONFIRMED → added to already_marked
     → PERMANENTLY removed from match_vault comparisons
     → CANNOT appear on any other face for rest of session
  2. Unrecognised faces always show "Unknown"
  3. Bolder, cleaner font (thickness=2, solid bg label)
  4. Threshold=0.45, Margin=0.10, Vote=4
  5. Cache keyed by face CENTER (60px grid) — no slot collision
  6. CSV uses VIDEO time not wall clock
  7. --start-time argument for real class time offset
  8. ✅ NEW: After register, face keeps showing NAME (green) not Unknown
"""

import cv2
import pickle
import csv
import numpy as np
import os
import math
import argparse
from insightface.app import FaceAnalysis
from ultralytics import YOLO
from datetime import datetime, timedelta
import torch
import time

VAULT_PATH     = r"data\biometric_vault.pkl"
THRESHOLD      = 0.45
MARGIN_MIN     = 0.10
MIN_FACE_PX    = 20
VOTE_THRESHOLD = 4
RE_EMBED_EVERY = 20
YOLO_IMG_SIZE  = 640
DISPLAY_W      = 1440
DISPLAY_H      = 810
SIDEBAR_W      = 270
TOPBAR_H       = 62
BOTBAR_H       = 130
VIDEO_AREA_W   = DISPLAY_W - SIDEBAR_W
GRID_CELL      = 60

C_WHITE  = (255, 255, 255)
C_GREEN  = ( 34, 197,  94)
C_RED    = ( 59,  68, 239)
C_AMBER  = (  0, 165, 255)
C_YELLOW = (  0, 220, 255)
C_GRAY   = (130, 130, 130)
FONT     = cv2.FONT_HERSHEY_DUPLEX
AA       = cv2.LINE_AA

print("=" * 60)
print("  VIT ELITE — process_video.py  (v5 + name-after-register fix)")
print("=" * 60)

if torch.cuda.is_available():
    YOLO_DEVICE  = 0
    IF_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    SKIP_FRAMES  = 5
    GPU_LABEL    = "GPU: CUDA"
    print(f"[GPU] {torch.cuda.get_device_name(0)}")
else:
    YOLO_DEVICE  = "cpu"
    IF_PROVIDERS = ["CPUExecutionProvider"]
    SKIP_FRAMES  = 20
    GPU_LABEL    = "CPU"
    print("[CPU] SKIP_FRAMES=20")

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


# ── VIDEO TIME HELPER ─────────────────────────────────────────
def frame_to_video_time(frame_no, src_fps, video_start_dt=None):
    if src_fps <= 0:
        src_fps = 25.0
    seconds = int(frame_no / src_fps)
    if video_start_dt:
        ts = video_start_dt + timedelta(seconds=seconds)
        return ts.strftime("%H:%M:%S")
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


# ── CSV SAVE ──────────────────────────────────────────────────
def save_csv(confirmed_present, video_date):
    path = f"data/attendance_{video_date}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Name", "Reg No", "Date", "Time", "Status"])
        for name, (reg, t) in sorted(confirmed_present.items()):
            w.writerow([name, reg, video_date, t, "Present"])
        for name, (stored, reg) in vault.items():
            if name not in confirmed_present:
                w.writerow([name, reg, video_date, "", "Absent"])
    print(f"[CSV] Saved → {path}")


# ── HELPERS ───────────────────────────────────────────────────
def emb_list(s):
    return s if isinstance(s, list) else [s]

def cosine(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / (d + 1e-9))

def safe_crop(frame, x1, y1, x2, y2, pad=0.20):
    H, W = frame.shape[:2]
    bw, bh = x2-x1, y2-y1
    if bw<=0 or bh<=0: return None
    px, py = int(bw*pad), int(bh*pad)
    cy1=max(0,y1-py); cy2=min(H,y2+py)
    cx1=max(0,x1-px); cx2=min(W,x2+px)
    if cy2<=cy1 or cx2<=cx1: return None
    crop = frame[cy1:cy2, cx1:cx2].copy()
    if crop is None or crop.size==0 or 0 in crop.shape: return None
    return crop

def get_embedding(crop_bgr):
    try:
        if crop_bgr is None: return None
        h, w = crop_bgr.shape[:2]
        if w<MIN_FACE_PX or h<MIN_FACE_PX: return None
        if w<112 or h<112:
            scale = max(112.0/w, 112.0/h) * 1.5
            nw=max(int(w*scale),112); nh=max(int(h*scale),112)
            crop_bgr = cv2.resize(crop_bgr,(nw,nh), interpolation=cv2.INTER_LANCZOS4)
            if crop_bgr is None or crop_bgr.size==0: return None
        rgb   = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        faces = face_app.get(rgb)
        if faces:
            best = max(faces, key=lambda f:(f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            if best.embedding is not None: return best.embedding
        if rec_model:
            r112 = cv2.resize(rgb,(112,112), interpolation=cv2.INTER_LINEAR)
            inp  = np.expand_dims(r112.astype(np.float32), 0)
            return rec_model.get_feat(inp).flatten()
        return None
    except: return None


def match_vault(vec):
    if not vault:
        return "Unknown", "", 0.0
    scores = []
    for n, (stored, reg) in vault.items():
        if n in already_marked:        # ← SKIP confirmed names FOREVER
            continue
        s = max(cosine(vec, e) for e in emb_list(stored))
        scores.append((s, n, reg))
    if not scores:
        return "Unknown", "", 0.0
    scores.sort(reverse=True)
    best_s, best_n, best_r = scores[0]
    if best_s < THRESHOLD:
        return "Unknown", "", best_s
    if len(scores) > 1:
        if (best_s - scores[1][0]) < MARGIN_MIN:
            return "Unknown", "", best_s
    return best_n, best_r, best_s


def face_cache_key(cx, cy):
    return ((cx // GRID_CELL) * GRID_CELL,
            (cy // GRID_CELL) * GRID_CELL)


# ── DISPLAY HELPERS ───────────────────────────────────────────
def letterbox_resize(frame, target_w, target_h):
    h, w = frame.shape[:2]
    scale = min(target_w/w, target_h/h)
    nw, nh = int(w*scale), int(h*scale)
    resized = cv2.resize(frame,(nw,nh), interpolation=cv2.INTER_LINEAR)
    pt=(target_h-nh)//2; pb=target_h-nh-pt
    pl=(target_w-nw)//2; pr=target_w-nw-pl
    return cv2.copyMakeBorder(resized,pt,pb,pl,pr,
                               cv2.BORDER_CONSTANT, value=(0,0,0))

def draw_grid(frame):
    H, W = frame.shape[:2]
    ov = frame.copy()
    for x in range(0,W,80): cv2.line(ov,(x,0),(x,H),(255,255,255),1,AA)
    for y in range(0,H,60): cv2.line(ov,(0,y),(W,y),(255,255,255),1,AA)
    cv2.addWeighted(ov,0.04,frame,0.96,0,frame)

def draw_scan_line(frame, tick):
    H, W = frame.shape[:2]
    y = int(abs(math.sin(tick*0.03))*H)
    cv2.line(frame,(0,y),(W,y),C_GREEN,1,AA)
    ov = frame.copy()
    for dy in [1,2,3]: cv2.line(ov,(0,y+dy),(W,y+dy),C_GREEN,1,AA)
    cv2.addWeighted(ov,0.05,frame,0.95,0,frame)

def draw_scan_ring(frame, cx, cy, rx, ry, angle, col):
    axes = (max(rx+16,20), max(ry+16,20))
    cv2.ellipse(frame,(cx,cy),axes,angle,0,270,col,2,AA)
    cv2.ellipse(frame,(cx,cy),axes,angle+180,0,90,col,2,AA)

def draw_face_box(frame, x1, y1, x2, y2, col):
    cv2.rectangle(frame,(x1,y1),(x2,y2),col,2,AA)
    L = max(12,min(22,(x2-x1)//4))
    for px,py,dx,dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame,(px,py),(px+dx*L,py),col,3,AA)
        cv2.line(frame,(px,py),(px,py+dy*L),col,3,AA)

def draw_label(frame, x1, y1, text, col):
    fs  = 0.55
    ft  = 2
    (tw,th),bl = cv2.getTextSize(text, FONT, fs, ft)
    pad = 8
    lx2 = x1 + tw + pad*2
    ly1 = max(0, y1 - th - bl - pad*2)
    cv2.rectangle(frame, (x1, ly1), (lx2, y1), col, cv2.FILLED)
    cv2.rectangle(frame, (x1, ly1), (lx2, y1), C_WHITE, 1, AA)
    cv2.putText(frame, text, (x1+pad, y1-bl-pad//2), FONT, fs, C_WHITE, ft, AA)

def draw_conf_bar(frame, x1, y2, x2, score):
    bw = x2-x1; filled = int(bw*min(score/0.6,1.0))
    cv2.rectangle(frame,(x1,y2+4),(x2,y2+10),(40,40,40),cv2.FILLED)
    col = C_GREEN if score>=THRESHOLD else C_RED
    cv2.rectangle(frame,(x1,y2+4),(x1+filled,y2+10),col,cv2.FILLED)

def draw_flash(frame, col, alpha=0.14):
    ov = frame.copy()
    cv2.rectangle(ov,(0,0),(frame.shape[1],frame.shape[0]),col,cv2.FILLED)
    cv2.addWeighted(ov,alpha,frame,1-alpha,0,frame)

def draw_top_bar(canvas, frame_no, total, fps, src_fps, video_time_str):
    H, W = canvas.shape[:2]
    ov = canvas.copy()
    cv2.rectangle(ov,(0,0),(W,TOPBAR_H),(8,8,12),cv2.FILLED)
    cv2.addWeighted(ov,0.88,canvas,0.12,0,canvas)
    cv2.line(canvas,(0,TOPBAR_H),(W,TOPBAR_H),(50,50,70),1,AA)
    cv2.putText(canvas,"VIT",(16,44),FONT,1.10,C_YELLOW,2,AA)
    cv2.putText(canvas,"CCTV ATTENDANCE ANALYSIS",(90,44),FONT,0.70,C_WHITE,2,AA)
    prog = f"{frame_no}/{total}" if total else str(frame_no)
    mid  = f"FRAME: {prog}  |  PROC: {fps:.1f} FPS  |  VIDEO TIME: {video_time_str}"
    (mw,_),_ = cv2.getTextSize(mid,FONT,0.55,1)
    cv2.putText(canvas,mid,(VIDEO_AREA_W//2-mw//2,44),FONT,0.55,C_WHITE,1,AA)
    clock = datetime.now().strftime("%I:%M:%S %p")
    (cw,_),_ = cv2.getTextSize(clock,FONT,0.65,1)
    cv2.putText(canvas,clock,(VIDEO_AREA_W-cw-16,44),FONT,0.65,C_WHITE,1,AA)
    (gw,_),_ = cv2.getTextSize(GPU_LABEL,FONT,0.48,1)
    cv2.putText(canvas,GPU_LABEL,(VIDEO_AREA_W-gw-16,22),FONT,0.48,C_GREEN,1,AA)

def draw_bottom_bar(canvas, n_present, n_absent, skip):
    H, W = canvas.shape[:2]
    ov = canvas.copy()
    cv2.rectangle(ov,(0,H-BOTBAR_H),(VIDEO_AREA_W,H),(8,8,12),cv2.FILLED)
    cv2.addWeighted(ov,0.88,canvas,0.12,0,canvas)
    cv2.line(canvas,(0,H-BOTBAR_H),(VIDEO_AREA_W,H-BOTBAR_H),(50,50,70),1,AA)
    cv2.putText(canvas,"PRESENT",(24,H-100),FONT,0.45,C_GRAY,1,AA)
    cv2.putText(canvas,str(n_present),(24,H-52),FONT,1.60,C_GREEN,2,AA)
    cv2.line(canvas,(220,H-118),(220,H-10),(50,50,70),1,AA)
    cv2.putText(canvas,"ABSENT",(234,H-100),FONT,0.45,C_GRAY,1,AA)
    cv2.putText(canvas,str(n_absent),(234,H-52),FONT,1.60,C_RED,2,AA)
    cv2.line(canvas,(430,H-118),(430,H-10),(50,50,70),1,AA)
    cv2.putText(canvas,"ENROLLED",(444,H-100),FONT,0.45,C_GRAY,1,AA)
    cv2.putText(canvas,str(len(vault)),(444,H-52),FONT,1.60,C_YELLOW,2,AA)
    cv2.putText(canvas,f"SKIP: 1/{skip}  [{GPU_LABEL}]",(650,H-14),FONT,0.48,C_GRAY,1,AA)
    cv2.putText(canvas,datetime.now().strftime("%d %B %Y"),(16,H-10),FONT,0.42,C_GRAY,1,AA)

def draw_sidebar(canvas, confirmed_present):
    H = canvas.shape[0]; px = VIDEO_AREA_W
    ov = canvas.copy()
    cv2.rectangle(ov,(px,0),(px+SIDEBAR_W,H),(10,10,16),cv2.FILLED)
    cv2.addWeighted(ov,0.90,canvas,0.10,0,canvas)
    cv2.line(canvas,(px,0),(px,H),(50,50,70),1,AA)
    cv2.putText(canvas,"ATTENDANCE",(px+10,36),FONT,0.55,C_YELLOW,2,AA)
    cv2.putText(canvas,"REGISTER",(px+10,58),FONT,0.50,C_YELLOW,2,AA)
    cv2.line(canvas,(px+5,66),(px+SIDEBAR_W-5,66),(50,50,70),1,AA)
    y = 92
    for name,(reg,ts) in list(confirmed_present.items())[:15]:
        cv2.circle(canvas,(px+14,y-6),8,C_GREEN,-1,AA)
        cv2.putText(canvas,f" {name[:17]}",(px+28,y),FONT,0.46,C_WHITE,2,AA)
        cv2.putText(canvas,f"  {reg}",(px+28,y+17),FONT,0.36,C_GRAY,1,AA)
        cv2.putText(canvas,f"  {ts}",(px+28,y+32),FONT,0.32,(100,100,100),1,AA)
        cv2.line(canvas,(px+10,y+38),(px+SIDEBAR_W-10,y+38),(30,30,40),1,AA)
        y += 54
        if y > H-150: break
    absent = len(vault)-len(confirmed_present)
    cv2.line(canvas,(px+5,H-130),(px+SIDEBAR_W-5,H-130),(50,50,70),1,AA)
    cv2.putText(canvas,f"Present : {len(confirmed_present)}",(px+10,H-105),FONT,0.50,C_GREEN,2,AA)
    cv2.putText(canvas,f"Absent  : {absent}",(px+10,H-75),FONT,0.50,C_RED,2,AA)
    cv2.putText(canvas,f"Enrolled: {len(vault)}",(px+10,H-45),FONT,0.50,C_YELLOW,2,AA)
    cv2.putText(canvas,"[Q] QUIT",(px+10,H-15),FONT,0.44,C_GRAY,1,AA)

def build_display(proc_frame, confirmed_present, frame_no,
                  total, fps_proc, src_fps, video_time_str):
    canvas = np.zeros((DISPLAY_H,DISPLAY_W,3), dtype=np.uint8)
    video_display = letterbox_resize(proc_frame, VIDEO_AREA_W, DISPLAY_H)
    canvas[:, :VIDEO_AREA_W] = video_display
    draw_sidebar(canvas, confirmed_present)
    draw_top_bar(canvas, frame_no, total, fps_proc, src_fps, video_time_str)
    draw_bottom_bar(canvas, len(confirmed_present),
                    len(vault)-len(confirmed_present), SKIP_FRAMES)
    return canvas


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def process_video(source, video_start_time_str=None):
    global already_marked
    already_marked = set()

    is_webcam = str(source).isdigit()
    video_start_dt = None
    video_date_str = datetime.now().strftime("%d-%m-%Y")

    if video_start_time_str and not is_webcam:
        try:
            fmt = "%H:%M:%S" if video_start_time_str.count(":") == 2 else "%H:%M"
            t   = datetime.strptime(video_start_time_str, fmt)
            video_start_dt = datetime.now().replace(
                hour=t.hour, minute=t.minute, second=t.second, microsecond=0)
            video_date_str = video_start_dt.strftime("%d-%m-%Y")
            print(f"[Video Start] {video_start_dt.strftime('%d-%m-%Y %H:%M:%S')}")
        except Exception as e:
            print(f"[WARN] Could not parse --start-time: {e}")

    if is_webcam:
        cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        total_frames = 0; src_fps = 30.0
    else:
        cap = cv2.VideoCapture(source)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source}"); return

    print(f"[Source] {source}  |  {total_frames} frames  |  {src_fps:.1f} FPS")

    cv2.namedWindow("VIT CCTV", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("VIT CCTV", DISPLAY_W, DISPLAY_H)

    pos_cache         = {}   # ✅ now stores "confirmed" flag too
    confirmed_present = {}
    vote_counts       = {}
    frame_no   = 0; proc_no = 0; scan_angle = 0
    flash_frames = 0; flash_col = C_GREEN; fps_proc = 0.0

    while True:
        ret, frame = cap.read()
        if not ret: print("[Done] End of source."); break

        frame_no += 1
        if frame_no % SKIP_FRAMES != 0: continue
        proc_no += 1
        t_proc = time.time()

        if is_webcam:
            video_time_str = datetime.now().strftime("%H:%M:%S")
        else:
            video_time_str = frame_to_video_time(frame_no, src_fps, video_start_dt)

        H, W = frame.shape[:2]
        draw_grid(frame)
        draw_scan_line(frame, frame_no)

        res = yolo.predict(frame, conf=0.40, verbose=False,
                           device=YOLO_DEVICE, imgsz=YOLO_IMG_SIZE)

        current_keys = set()
        detected = []
        if res[0].boxes and len(res[0].boxes):
            for b in res[0].boxes:
                x1,y1,x2,y2 = map(int, b.xyxy[0].cpu().numpy())
                detected.append((x1,y1,x2,y2))
                cx = (x1+x2)//2; cy = (y1+y2)//2
                current_keys.add(face_cache_key(cx, cy))

        pos_cache = {k:v for k,v in pos_cache.items() if k in current_keys}

        for (x1,y1,x2,y2) in detected:
            bw, bh = x2-x1, y2-y1
            cx = (x1+x2)//2; cy = (y1+y2)//2
            key    = face_cache_key(cx, cy)
            cached = pos_cache.get(key)

            # ✅ FIX: if slot already confirmed, keep showing name — no re-embed
            if cached and cached.get("confirmed", False):
                name  = cached["name"]
                reg   = cached["reg"]
                score = cached["score"]
            else:
                needs = (
                    cached is None or
                    (proc_no - cached.get("last", 0)) >= RE_EMBED_EVERY or
                    (cached.get("name", "Unknown") in already_marked
                     and not cached.get("confirmed", False))
                )
                name, reg, score = "Unknown", "", 0.0

                if needs:
                    crop = safe_crop(frame, x1, y1, x2, y2)
                    if crop is not None:
                        emb = get_embedding(crop)
                        if emb is not None:
                            name, reg, score = match_vault(emb)
                            confirmed_flag = name in confirmed_present
                            pos_cache[key] = {
                                "name": name, "reg": reg,
                                "score": score, "last": proc_no,
                                "confirmed": confirmed_flag
                            }
                            if name != "Unknown":
                                vote_counts[name] = vote_counts.get(name,0) + 1
                                if (vote_counts[name] >= VOTE_THRESHOLD
                                        and name not in confirmed_present):
                                    confirmed_present[name] = (reg, video_time_str)
                                    already_marked.add(name)
                                    # ✅ mark slot so future frames show name
                                    pos_cache[key]["confirmed"] = True
                                    flash_col = C_GREEN; flash_frames = 8
                                    print(f"  PRESENT: {name} ({reg}) @ {video_time_str}")
                else:
                    name  = cached["name"]
                    reg   = cached["reg"]
                    score = cached["score"]

            # ── LABEL & COLOUR ────────────────────────────────
            if name != "Unknown" and name in confirmed_present:
                col = C_GREEN
                lbl = f"{name}  {reg}"       # ✅ always shows name after register
            elif name != "Unknown":
                col = C_AMBER
                lbl = f"{name}  [{vote_counts.get(name,0)}/{VOTE_THRESHOLD}]"
            else:
                col = C_RED
                lbl = "Unknown"

            draw_face_box(frame, x1, y1, x2, y2, col)
            draw_scan_ring(frame, cx, cy, bw//2, bh//2, scan_angle, col)
            draw_label(frame, x1, y1, lbl, col)
            draw_conf_bar(frame, x1, y2, x2, score)

        fps_proc = 1.0 / (time.time()-t_proc + 1e-6)
        if flash_frames > 0: draw_flash(frame, flash_col); flash_frames -= 1
        scan_angle = (scan_angle + 4) % 360

        canvas = build_display(frame, confirmed_present, frame_no,
                               total_frames, fps_proc, src_fps, video_time_str)
        cv2.imshow("VIT CCTV", canvas)
        if cv2.waitKey(1) & 0xFF == ord("q"): break

    cap.release()
    cv2.destroyAllWindows()
    save_csv(confirmed_present, video_date_str)
    print(f"[Summary] Present: {len(confirmed_present)} / {len(vault)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",
                        default=r"C:\Users\SAAD\Videos\Captures\PRP G03 sample 2.mp4",
                        help="Video file path or 0 for webcam")
    parser.add_argument("--start-time", default=None,
                        help="Real-world start time e.g. 12:00 or 12:00:00")
    args = parser.parse_args()
    process_video(args.source, args.start_time)