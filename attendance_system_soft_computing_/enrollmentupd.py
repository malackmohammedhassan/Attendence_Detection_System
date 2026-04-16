"""
VIT Attendance System — ENROLLMENT (Multi-Person + 5-Angle)
Fixed:
  1. MATCH_THRESHOLD raised to 0.38 (no more false matches)
  2. match_vault checks margin between 1st and 2nd best score
  3. draw_person_badge shows ONLY green (confirmed) or red (Unknown)
  4. No orange "matching" label that causes confusion
"""

import cv2
import os
import pickle
import numpy as np
from insightface.app import FaceAnalysis
from ultralytics import YOLO
import torch

# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════
VAULT_PATH      = r"data/biometric_vault.pkl"
MATCH_THRESHOLD = 0.38        # FIX 1: raised from 0.20 → 0.38
MIN_FACE_PX     = 20
MAX_PHOTOS      = 5
MARGIN_MIN      = 0.08        # FIX 2: min gap between 1st and 2nd best match

ANGLES = ["FRONT", "LEFT", "RIGHT", "TOP", "BOTTOM"]

# ══════════════════════════════════════════════════════════════
#  COLOURS
# ══════════════════════════════════════════════════════════════
C_GREEN  = ( 34, 197,  94)
C_RED    = ( 59,  68, 239)
C_YELLOW = (  0, 220, 255)
C_WHITE  = (255, 255, 255)
C_BLACK  = (  0,   0,   0)
C_DARK   = ( 15,  15,  20)
C_ORANGE = (  0, 165, 255)
C_CYAN   = (255, 200,   0)
C_GRAY   = (130, 130, 130)
FONT     = cv2.FONT_HERSHEY_DUPLEX
AA       = cv2.LINE_AA

# ══════════════════════════════════════════════════════════════
#  GPU
# ══════════════════════════════════════════════════════════════
print("="*60)
print("  VIT SMART ENROLLMENT  —  5-Angle Multi-Person  (Fixed)")
print("="*60)

if torch.cuda.is_available():
    YOLO_DEVICE  = 0
    IF_PROVIDERS = ["CUDAExecutionProvider","CPUExecutionProvider"]
    print(f"[GPU] {torch.cuda.get_device_name(0)}")
else:
    YOLO_DEVICE  = "cpu"
    IF_PROVIDERS = ["CPUExecutionProvider"]
    print("[CPU] mode")

# ══════════════════════════════════════════════════════════════
#  MODELS
# ══════════════════════════════════════════════════════════════
yolo = YOLO(r"models/yolov8n-face.pt")
print("[YOLOv8] Loaded.")

face_app = FaceAnalysis(name="buffalo_l", providers=IF_PROVIDERS)
face_app.prepare(ctx_id=0, det_size=(640,640))
print("[InsightFace] buffalo_l ready.")

rec_model = face_app.models.get("recognition")
print("[ArcFace] Ready." if rec_model else "[ArcFace] WARNING: not found!")

# ══════════════════════════════════════════════════════════════
#  VAULT
# ══════════════════════════════════════════════════════════════
os.makedirs("data", exist_ok=True)
if os.path.exists(VAULT_PATH):
    with open(VAULT_PATH,"rb") as f:
        vault = pickle.load(f)
    print(f"[Vault] {len(vault)} student(s) loaded.")
else:
    vault = {}
    print("[Vault] Empty — starting fresh.")

# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════
def emb_list(s):
    return s if isinstance(s, list) else [s]

def cosine(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / (d + 1e-9))

def safe_crop(frame, x1, y1, x2, y2, pad=0.25):
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
            if best.embedding is not None:
                return best.embedding
        if rec_model:
            r112 = cv2.resize(rgb,(112,112), interpolation=cv2.INTER_LINEAR)
            inp  = np.expand_dims(r112.astype(np.float32), 0)
            return rec_model.get_feat(inp).flatten()
        return None
    except: return None


# ══════════════════════════════════════════════════════════════
#  FIX 1+2: STRICT match_vault — threshold + margin check
# ══════════════════════════════════════════════════════════════
def match_vault(vec):
    if not vault:
        return "Unknown", "", 0.0

    scores = []
    for n, (stored, reg) in vault.items():
        s = max(cosine(vec, e) for e in emb_list(stored))
        scores.append((s, n, reg))

    scores.sort(reverse=True)
    best_s, best_n, best_r = scores[0]

    # Must exceed hard threshold
    if best_s < MATCH_THRESHOLD:
        return "Unknown", "", best_s

    # Must be clearly better than 2nd best (margin check)
    # Prevents A's face showing on B when they look similar
    if len(scores) > 1:
        second_s = scores[1][0]
        if (best_s - second_s) < MARGIN_MIN:
            return "Unknown", "", best_s

    return best_n, best_r, best_s


def save_vault():
    with open(VAULT_PATH,"wb") as f:
        pickle.dump(vault, f)

# ══════════════════════════════════════════════════════════════
#  ANGLE CAPTURE SCREEN
# ══════════════════════════════════════════════════════════════
ANGLE_GUIDE = {
    "FRONT":  "Look straight at camera",
    "LEFT":   "Turn head to your LEFT",
    "RIGHT":  "Turn head to your RIGHT",
    "TOP":    "Tilt head slightly UPWARD",
    "BOTTOM": "Tilt head slightly DOWNWARD",
}

def capture_angles(cap, name, regno):
    """Full-screen guided 5-angle capture. Returns list of embeddings."""
    embeddings = []
    angle_idx  = 0

    while angle_idx < len(ANGLES):
        ret, frame = cap.read()
        if not ret: break
        H, W = frame.shape[:2]

        angle = ANGLES[angle_idx]
        guide = ANGLE_GUIDE[angle]
        progress = f"{angle_idx}/{len(ANGLES)} angles done"

        res = yolo.predict(frame, conf=0.5, verbose=False, device=YOLO_DEVICE)
        face_box = None
        if res[0].boxes and len(res[0].boxes):
            best_b = max(res[0].boxes,
                         key=lambda b: (int(b.xyxy[0][2])-int(b.xyxy[0][0]))*
                                       (int(b.xyxy[0][3])-int(b.xyxy[0][1])))
            x1,y1,x2,y2 = map(int, best_b.xyxy[0].cpu().numpy())
            face_box = (x1,y1,x2,y2)

            cx=(x1+x2)//2; cy=(y1+y2)//2
            rw=(x2-x1)//2+20; rh=(y2-y1)//2+20
            cv2.ellipse(frame,(cx,cy),(rw,rh),0,0,360,C_GREEN,3,AA)
            cv2.rectangle(frame,(x1,y1),(x2,y2),C_GREEN,2,AA)

        ov = frame.copy()
        cv2.rectangle(ov,(0,0),(W,56),(8,8,8),cv2.FILLED)
        cv2.addWeighted(ov,0.85,frame,0.15,0,frame)
        cv2.putText(frame,f"ENROLLING: {name}  [{regno}]",(12,34),FONT,0.75,C_YELLOW,1,AA)
        cv2.putText(frame,progress,(W-280,34),FONT,0.60,C_GRAY,1,AA)

        ov2 = frame.copy()
        cv2.rectangle(ov2,(0,H-130),(W,H),(8,8,8),cv2.FILLED)
        cv2.addWeighted(ov2,0.85,frame,0.15,0,frame)

        dot_y = H-100; dot_spacing = 60
        start_x = W//2 - (len(ANGLES)*dot_spacing)//2
        for i, ang in enumerate(ANGLES):
            cx_dot = start_x + i*dot_spacing
            if i < angle_idx:
                cv2.circle(frame,(cx_dot,dot_y),14,C_GREEN,-1,AA)
                cv2.putText(frame,"v",(cx_dot-6,dot_y+6),FONT,0.45,C_BLACK,2,AA)
            elif i == angle_idx:
                cv2.circle(frame,(cx_dot,dot_y),16,C_YELLOW,3,AA)
                cv2.circle(frame,(cx_dot,dot_y),10,C_YELLOW,-1,AA)
            else:
                cv2.circle(frame,(cx_dot,dot_y),14,(60,60,60),-1,AA)
            (lw,_),_ = cv2.getTextSize(ang,FONT,0.38,1)
            cv2.putText(frame,ang,(cx_dot-lw//2,dot_y+30),
                        FONT,0.38,C_WHITE if i==angle_idx else C_GRAY,1,AA)

        cv2.putText(frame,f"Step {angle_idx+1}/5:  {guide}",
                    (W//2-300,H-42),FONT,0.70,C_WHITE,1,AA)
        cv2.putText(frame,"Press  [SPACE]  to capture     [ESC] to cancel",
                    (W//2-300,H-14),FONT,0.55,C_CYAN,1,AA)

        display = cv2.resize(frame,(1280,720))
        cv2.imshow("VIT Smart Enrollment", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" ") or key == 13:
            if face_box:
                x1,y1,x2,y2 = face_box
                crop = safe_crop(frame,x1,y1,x2,y2)
                emb  = get_embedding(crop) if crop is not None else None
                if emb is not None:
                    embeddings.append(emb)
                    angle_idx += 1
                    print(f"  [{angle}] captured ({angle_idx}/{len(ANGLES)})")
                else:
                    flash = frame.copy()
                    cv2.rectangle(flash,(0,0),(W,H),(0,0,200),cv2.FILLED)
                    cv2.addWeighted(flash,0.25,frame,0.75,0,frame)
                    cv2.putText(frame,"Face not detected clearly — try again",
                                (W//2-280,H//2),FONT,0.80,C_WHITE,2,AA)
                    cv2.imshow("VIT Smart Enrollment", cv2.resize(frame,(1280,720)))
                    cv2.waitKey(800)
            else:
                cv2.putText(frame,"No face found — position yourself clearly",
                            (W//2-300,H//2),FONT,0.75,C_RED,2,AA)
                cv2.imshow("VIT Smart Enrollment", cv2.resize(frame,(1280,720)))
                cv2.waitKey(800)

        elif key == 27:
            print("  [Cancelled]")
            return None

    return embeddings if len(embeddings)==len(ANGLES) else None

# ══════════════════════════════════════════════════════════════
#  FIX 3: draw_person_badge — GREEN (matched) or RED (unknown) only
# ══════════════════════════════════════════════════════════════
def draw_person_badge(frame, x1, y1, x2, y2, pid, name, reg, score, enrolled):
    # FIX: removed orange "matching" — only confirmed green or unknown red
    if name != "Unknown" and enrolled:
        col = C_GREEN
        lbl = f"{name}  {reg}  ({score:.2f})"
    else:
        col = C_RED
        lbl = f"Unknown  ({score:.2f})"

    cv2.rectangle(frame,(x1,y1),(x2,y2),col,2,AA)
    L = max(8,min(16,(x2-x1)//4))
    for (px,py,dx,dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame,(px,py),(px+dx*L,py),col,3,AA)
        cv2.line(frame,(px,py),(px,py+dy*L),col,3,AA)

    badge_cx = x1+(x2-x1)//2
    badge_cy = max(28, y1-22)
    cv2.circle(frame,(badge_cx,badge_cy),22,col,-1,AA)
    cv2.circle(frame,(badge_cx,badge_cy),22,C_WHITE,2,AA)
    pid_txt = str(pid)
    (pw,ph),_ = cv2.getTextSize(pid_txt,FONT,0.75,2)
    cv2.putText(frame,pid_txt,(badge_cx-pw//2,badge_cy+ph//2),FONT,0.75,C_BLACK,2,AA)

    fs=0.50
    (tw,th),bl = cv2.getTextSize(lbl,FONT,fs,1)
    pad=5; lx1=x1; lx2=x1+tw+pad*2
    ly1=max(0,y1-th-bl-pad*2); ly2=y1
    ov = frame.copy()
    cv2.rectangle(ov,(lx1,ly1),(lx2,ly2),col,cv2.FILLED)
    cv2.addWeighted(ov,0.70,frame,0.30,0,frame)
    cv2.putText(frame,lbl,(lx1+pad,ly2-bl-pad//2),FONT,fs,C_WHITE,1,AA)

def draw_top_bar(frame, n_vault):
    H,W = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov,(0,0),(W,48),(8,8,8),cv2.FILLED)
    cv2.addWeighted(ov,0.85,frame,0.15,0,frame)
    cv2.line(frame,(0,48),(W,48),(50,50,50),1,AA)
    gpu = "GPU: CUDA" if torch.cuda.is_available() else "CPU"
    cv2.putText(frame,"VIT SMART ENROLLMENT",(12,32),FONT,0.75,C_YELLOW,1,AA)
    mid = f"Enrolled in vault: {n_vault} students"
    (mw,_),_ = cv2.getTextSize(mid,FONT,0.60,1)
    cv2.putText(frame,mid,(W//2-mw//2,32),FONT,0.60,C_WHITE,1,AA)
    (gw,_),_ = cv2.getTextSize(gpu,FONT,0.55,1)
    cv2.putText(frame,gpu,(W-gw-12,32),FONT,0.55,C_GREEN,1,AA)

def draw_instruction_bar(frame):
    H,W = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov,(0,H-38),(W,H),(8,8,8),cv2.FILLED)
    cv2.addWeighted(ov,0.85,frame,0.15,0,frame)
    cv2.line(frame,(0,H-38),(W,H-38),(50,50,50),1,AA)
    cv2.putText(frame,
        "  [1-9] Select person to enroll    [Q] Quit",
        (10,H-12),FONT,0.55,(200,200,200),1,AA)

def draw_vault_sidebar(frame):
    H,W = frame.shape[:2]
    SW=260; px=W-SW
    ov = frame.copy()
    cv2.rectangle(ov,(px,48),(W,H),(12,12,20),cv2.FILLED)
    cv2.addWeighted(ov,0.82,frame,0.18,0,frame)
    cv2.line(frame,(px,48),(px,H),(55,55,75),1,AA)
    cv2.putText(frame,"VAULT",(px+10,78),FONT,0.60,C_YELLOW,1,AA)
    cv2.putText(frame,f"({len(vault)} enrolled)",(px+80,78),FONT,0.50,C_GRAY,1,AA)
    cv2.line(frame,(px+5,84),(W-5,84),(55,55,75),1,AA)
    y=108
    for i,(n,(stored,reg)) in enumerate(list(vault.items())[:14]):
        c   = len(emb_list(stored))
        bar = "█"*c + "░"*(MAX_PHOTOS-c)
        cv2.putText(frame,f"  {n[:16]}",(px+5,y),FONT,0.44,C_WHITE,1,AA)
        cv2.putText(frame,f"  {reg}  [{bar}]",(px+5,y+16),FONT,0.34,C_GRAY,1,AA)
        y+=38

def show_success(cap, name, regno, n_photos):
    for _ in range(60):
        ret, frame = cap.read()
        if not ret: break
        H,W = frame.shape[:2]
        ov = frame.copy()
        cv2.rectangle(ov,(0,0),(W,H),(10,40,10),cv2.FILLED)
        cv2.addWeighted(ov,0.45,frame,0.55,0,frame)
        msgs = [
            ("ENROLLMENT SUCCESSFUL",            0.00, C_GREEN,  1.2, 2),
            (f"Name   :  {name}",                0.15, C_WHITE,  0.75,1),
            (f"Reg No :  {regno}",               0.25, C_YELLOW, 0.75,1),
            (f"Photos :  {n_photos}/5 angles",   0.35, C_WHITE,  0.65,1),
            ("Continuing in 2s...",               0.50, C_GRAY,   0.50,1),
        ]
        for txt,rel_y,col,fs,ft in msgs:
            (tw,th),_ = cv2.getTextSize(txt,FONT,fs,ft)
            y = int(H*0.30 + H*rel_y)
            cv2.putText(frame,txt,(W//2-tw//2,y),FONT,fs,col,ft,AA)
        cv2.imshow("VIT Smart Enrollment", cv2.resize(frame,(1280,720)))
        cv2.waitKey(33)

# ══════════════════════════════════════════════════════════════
#  CAMERA
# ══════════════════════════════════════════════════════════════
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

cv2.namedWindow("VIT Smart Enrollment", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("VIT Smart Enrollment",
                      cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# ══════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════
while True:
    ret, frame = cap.read()
    if not ret: break
    H, W = frame.shape[:2]

    res = yolo.predict(frame, conf=0.50, verbose=False, device=YOLO_DEVICE)
    boxes = []
    if res[0].boxes and len(res[0].boxes):
        sorted_b = sorted(res[0].boxes,
                          key=lambda b: int(b.xyxy[0][0].item()))
        for b in sorted_b[:9]:
            x1,y1,x2,y2 = map(int, b.xyxy[0].cpu().numpy())
            boxes.append((x1,y1,x2,y2))

    results = []
    for (x1,y1,x2,y2) in boxes:
        crop = safe_crop(frame,x1,y1,x2,y2)
        emb  = get_embedding(crop) if crop is not None else None
        if emb is not None:
            name,reg,score = match_vault(emb)
            enrolled = (name in vault)
        else:
            name,reg,score,enrolled = "Unknown","",0.0,False
        results.append((name,reg,score,enrolled))

    for i,((x1,y1,x2,y2),(name,reg,score,enrolled)) in enumerate(
            zip(boxes,results)):
        draw_person_badge(frame,x1,y1,x2,y2,i+1,name,reg,score,enrolled)

    draw_vault_sidebar(frame)
    draw_top_bar(frame, len(vault))
    draw_instruction_bar(frame)

    display = cv2.resize(frame,(1280,720))
    cv2.imshow("VIT Smart Enrollment", display)
    key = cv2.waitKey(1) & 0xFF

    if ord("1") <= key <= ord("9"):
        idx = key - ord("1")
        if idx < len(boxes):
            person_no = idx+1
            print(f"\n[Select] Person {person_no} selected for enrollment")
            nm  = input(f"  Name for Person {person_no}  : ").strip()
            reg = input(f"  Registration No          : ").strip()
            if not nm or not reg:
                print("  [!] Name and Reg required."); continue
            print(f"  [Capture] Starting 5-angle capture for {nm}...")
            embeddings = capture_angles(cap, nm, reg)
            if embeddings and len(embeddings) == 5:
                vault[nm] = (embeddings, reg)
                save_vault()
                print(f"  [Saved] {nm} ({reg}) — 5 angles enrolled.")
                show_success(cap, nm, reg, 5)
            else:
                print(f"  [!] Enrollment cancelled or incomplete.")
        else:
            print(f"  [!] No Person {idx+1} detected.")

    elif key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("[Done] Enrollment session closed.")
print(f"[Vault] {len(vault)} student(s) enrolled total.")