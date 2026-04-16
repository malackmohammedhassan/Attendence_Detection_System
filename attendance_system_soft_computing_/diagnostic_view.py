import os, cv2, pickle, torch
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace

# --- 1. MANDATORY SECURITY BYPASS ---
os.environ['TORCH_DYNAMO_DISABLE'] = '1'
os.environ['PYTHONOPTIMIZE'] = '1'

# Initialize Models
model = YOLO(r'models\yolov8n-face.pt')
vault_path = r'data\biometric_vault.pkl'

def run_diagnostics():
    if not os.path.exists(vault_path):
        print("Error: No biometric_vault.pkl found. Please enroll first.")
        return

    with open(vault_path, 'rb') as f:
        vault = pickle.load(f)

    cap = cv2.VideoCapture(0)
    print("DIAGNOSTIC MODE ACTIVE: Press 'Q' to Quit.")

    while True:
        ret, frame = cap.read()
        if not ret: continue
        
        results = model.predict(frame, conf=0.7, verbose=False)
        
        # UI Header
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (50, 50, 50), -1)
        cv2.putText(frame, "AI DIAGNOSTIC FEED - ANALYZING DISTANCE", (10, 25), 0, 0.6, (255, 255, 255), 1)

        if results[0].boxes:
            box = results[0].boxes.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, box)
            
            try:
                # Extract ArcFace Embedding
                face_crop = frame[y1:y2, x1:x2]
                res = DeepFace.represent(face_crop, model_name="ArcFace", enforce_detection=False)[0]
                live_vec = np.array(res["embedding"])
                
                best_dist = 1.0
                closest_name = "None"

                # Calculate distance for every person in your database
                for name, (stored_vec, reg_no) in vault.items():
                    # Cosine Distance Formula
                    dist = 1 - (np.dot(live_vec, stored_vec)/(np.linalg.norm(live_vec)*np.linalg.norm(stored_vec)))
                    
                    # Track the closest match for the UI
                    if dist < best_dist:
                        best_dist = dist
                        closest_name = name

                # Color logic based on yesterday's success (0.55 threshold)
                color = (0, 255, 0) if best_dist < 0.55 else (0, 0, 255)
                
                # Draw Box and Live Data
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Display the "Math" on the screen
                cv2.putText(frame, f"Closest: {closest_name}", (x1, y2 + 25), 0, 0.6, color, 2)
                cv2.putText(frame, f"Dist: {best_dist:.4f}", (x1, y2 + 50), 0, 0.6, color, 2)
                cv2.putText(frame, "Target: < 0.5500", (x1, y2 + 75), 0, 0.5, (255, 255, 255), 1)

            except Exception as e:
                cv2.putText(frame, "AI Processing...", (x1, y1 - 10), 0, 0.5, (255, 255, 0), 1)

        cv2.imshow("VIT AI Diagnostic Tool", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    run_diagnostics()