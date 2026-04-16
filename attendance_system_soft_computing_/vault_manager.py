"""
VIT Attendance System — VAULT MANAGER
════════════════════════════════════════════════════════════════
View, preview and remove enrolled students from biometric_vault.pkl

CONTROLS (in preview window)
  [D]   Delete selected student permanently
  [N]   Next student
  [P]   Previous student
  [ESC] Close
"""

import cv2
import os
import pickle
import numpy as np
from insightface.app import FaceAnalysis
import torch

# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════
VAULT_PATH = r"data/biometric_vault.pkl"

C_GREEN  = ( 34, 197,  94)
C_RED    = ( 59,  68, 239)
C_YELLOW = (  0, 220, 255)
C_WHITE  = (255, 255, 255)
C_BLACK  = (  0,   0,   0)
C_DARK   = ( 20,  20,  20)
C_ORANGE = (  0, 165, 255)
C_GRAY   = (120, 120, 120)
FONT     = cv2.FONT_HERSHEY_DUPLEX
AA       = cv2.LINE_AA

CARD_W, CARD_H = 900, 500

# ══════════════════════════════════════════════════════════════
#  LOAD VAULT
# ══════════════════════════════════════════════════════════════
def load_vault():
    if not os.path.exists(VAULT_PATH):
        print(f"[ERROR] Vault not found: {VAULT_PATH}")
        print("        Run enrollment.py first.")
        exit(1)
    with open(VAULT_PATH, "rb") as f:
        return pickle.load(f)

def save_vault(vault):
    with open(VAULT_PATH, "wb") as f:
        pickle.dump(vault, f)
    print(f"[Vault] Saved — {len(vault)} student(s) remaining.")

# ══════════════════════════════════════════════════════════════
#  TERMINAL LISTING
# ══════════════════════════════════════════════════════════════
def print_vault(vault):
    print("\n" + "="*65)
    print(f"  BIOMETRIC VAULT  —  {len(vault)} student(s) enrolled")
    print("="*65)
    if not vault:
        print("  (empty)")
    for i, (name, (stored, reg)) in enumerate(vault.items(), 1):
        stored = stored if isinstance(stored, list) else [stored]
        c   = len(stored)
        bar = "█"*c + "░"*(5-c)
        print(f"  {i:>3}. {name:<28} {reg:<16} [{bar}] {c} photo(s)")
    print("="*65 + "\n")

# ══════════════════════════════════════════════════════════════
#  DRAW STUDENT CARD
# ══════════════════════════════════════════════════════════════
def make_card(idx, name, stored, reg, total):
    stored = stored if isinstance(stored, list) else [stored]
    n_photos = len(stored)

    card = np.zeros((CARD_H, CARD_W, 3), dtype=np.uint8)
    card[:] = (25, 25, 35)

    # Header bar
    cv2.rectangle(card, (0,0), (CARD_W, 60), (40,40,60), cv2.FILLED)
    cv2.putText(card, "VIT VAULT MANAGER", (20,40), FONT, 0.7, C_YELLOW, 1, AA)
    nav = f"Student {idx+1} of {total}"
    (nw,_),_ = cv2.getTextSize(nav, FONT, 0.55, 1)
    cv2.putText(card, nav, (CARD_W-nw-20, 38), FONT, 0.55, C_GRAY, 1, AA)

    # Student info block
    cv2.putText(card, "NAME",    (30, 105), FONT, 0.42, C_GRAY,  1, AA)
    cv2.putText(card, name,      (30, 135), FONT, 0.80, C_WHITE, 1, AA)
    cv2.putText(card, "REG NO",  (30, 175), FONT, 0.42, C_GRAY,  1, AA)
    cv2.putText(card, reg,       (30, 205), FONT, 0.75, C_YELLOW,1, AA)
    cv2.putText(card, "PHOTOS",  (30, 245), FONT, 0.42, C_GRAY,  1, AA)
    bar = "█"*n_photos + "░"*(5-n_photos)
    cv2.putText(card, f"{bar}  {n_photos}/5", (30,275), FONT, 0.65, C_GREEN,1, AA)

    # Embedding dimension info
    cv2.putText(card, "EMBEDDING", (30, 315), FONT, 0.42, C_GRAY, 1, AA)
    cv2.putText(card, f"ArcFace  512-dim  x{n_photos} angles",
                (30, 342), FONT, 0.50, C_WHITE, 1, AA)

    # Divider
    cv2.line(card, (20, 365), (CARD_W-20, 365), (60,60,80), 1, AA)

    # Controls
    controls = [
        ("[D]", "Delete this student", C_RED),
        ("[N]", "Next student",        C_GREEN),
        ("[P]", "Previous student",    C_GREEN),
        ("[ESC]","Close",              C_GRAY),
    ]
    cx = 30
    for key, desc, col in controls:
        cv2.putText(card, key,  (cx, 410), FONT, 0.55, col,    2, AA)
        (kw,_),_ = cv2.getTextSize(key, FONT, 0.55, 2)
        cv2.putText(card, desc, (cx+kw+8, 410), FONT, 0.45, C_GRAY, 1, AA)
        (dw,_),_ = cv2.getTextSize(desc, FONT, 0.45, 1)
        cx += kw + dw + 28

    # Delete warning
    cv2.putText(card,
                "WARNING: [D] permanently removes student from vault",
                (30, 455), FONT, 0.40, (80,80,200), 1, AA)

    # Right panel — embedding visualisation bars
    bx = 560
    cv2.putText(card, "EMBEDDING SNAPSHOT", (bx, 105), FONT, 0.42, C_GRAY, 1, AA)
    emb = stored[0]
    emb_norm = (emb - emb.min()) / (emb.max() - emb.min() + 1e-9)
    bar_h = 180
    n_bars = 64
    bw_each = (CARD_W - bx - 20) // n_bars
    for j in range(n_bars):
        val = float(emb_norm[j * (512 // n_bars)])
        bh  = int(val * bar_h)
        col_b = (
            int(34  + (0-34)  * val),
            int(197 + (165-197)* val),
            int(94  + (255-94) * val)
        )
        bx2 = bx + j*bw_each
        cv2.rectangle(card,
                      (bx2, 120 + bar_h - bh),
                      (bx2 + bw_each-1, 120 + bar_h),
                      col_b, cv2.FILLED)

    cv2.putText(card, "First 64 dims of ArcFace vector",
                (bx, 325), FONT, 0.38, C_GRAY, 1, AA)

    return card

# ══════════════════════════════════════════════════════════════
#  CONFIRM DELETE CARD
# ══════════════════════════════════════════════════════════════
def make_confirm_card(name, reg):
    card = np.zeros((CARD_H, CARD_W, 3), dtype=np.uint8)
    card[:] = (35, 20, 20)
    cv2.rectangle(card, (0,0), (CARD_W,60), (80,20,20), cv2.FILLED)
    cv2.putText(card, "CONFIRM DELETE", (20,40), FONT, 0.75, C_RED, 2, AA)
    cv2.putText(card, "Are you sure you want to remove:",
                (30, 110), FONT, 0.55, C_WHITE, 1, AA)
    cv2.putText(card, name, (30, 170), FONT, 1.0, C_YELLOW, 2, AA)
    cv2.putText(card, reg,  (30, 220), FONT, 0.75, C_WHITE,  1, AA)
    cv2.putText(card, "This cannot be undone.",
                (30, 280), FONT, 0.55, (100,100,255), 1, AA)
    cv2.line(card, (20,310),(CARD_W-20,310),(80,40,40),1,AA)
    cv2.putText(card, "[Y]  Yes, delete permanently",
                (30, 370), FONT, 0.65, C_RED, 1, AA)
    cv2.putText(card, "[N]  No, keep student",
                (30, 420), FONT, 0.65, C_GREEN, 1, AA)
    return card

# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    vault = load_vault()
    print_vault(vault)

    if not vault:
        print("[Vault] No students enrolled yet.")
        return

    cv2.namedWindow("VIT Vault Manager", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("VIT Vault Manager", CARD_W, CARD_H)

    names  = list(vault.keys())
    idx    = 0

    while True:
        if not vault:
            print("[Vault] All students removed — vault is empty.")
            break

        names = list(vault.keys())
        idx   = min(idx, len(names)-1)
        name  = names[idx]
        stored, reg = vault[name]

        card = make_card(idx, name, stored, reg, len(names))
        cv2.imshow("VIT Vault Manager", card)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("n") or key == 83:   # N or Right arrow
            idx = (idx + 1) % len(names)

        elif key == ord("p") or key == 81: # P or Left arrow
            idx = (idx - 1) % len(names)

        elif key == ord("d"):
            # Show confirm screen
            confirm = make_confirm_card(name, reg)
            cv2.imshow("VIT Vault Manager", confirm)
            c = cv2.waitKey(0) & 0xFF
            if c == ord("y"):
                del vault[name]
                save_vault(vault)
                print(f"  [Deleted] {name} ({reg}) removed from vault.")
                print_vault(vault)
                if idx >= len(vault):
                    idx = max(0, len(vault)-1)
            else:
                print(f"  [Cancelled] {name} kept.")

        elif key == 27 or key == ord("q"):  # ESC or Q
            break

    cv2.destroyAllWindows()
    print("[Vault Manager] Closed.")
    print_vault(vault)

if __name__ == "__main__":
    main()
