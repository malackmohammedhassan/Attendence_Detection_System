# Run Guide: Attendance System (Desktop)

This guide is for the desktop attendance module inside attendance*system_soft_computing*.

## Project Path

attendance*system_soft_computing*

## 1. Prerequisites

- Windows 10/11
- Python 3.9+ (3.10 or 3.11 recommended)
- Webcam for live mode
- Optional NVIDIA GPU (faster processing)

## 2. Install Dependencies

Open terminal in attendance*system_soft_computing* and run:

```bash
pip install -r requirements2.txt
```

## 3. Enroll Students (Required First)

Create biometric records before attendance:

```bash
python enrollmentupd.py
```

Output created:

- data/biometric_vault.pkl

## 4. Run Live Attendance (Kiosk Mode)

For real-time webcam attendance:

```bash
python main.py
```

Live output:

- data/live_attendance.csv

## 5. Run CCTV / Video Attendance (Offline or Live)

### A. Process a recorded video

```bash
python process_videoupd.py --source "C:\path\to\class_video.mp4"
```

### B. Process recorded video with real class start time

```bash
python process_videoupd.py --source "C:\path\to\class_video.mp4" --start-time 09:00
```

### C. Multi-face webcam mode

```bash
python process_videoupd.py --source 0
```

Output created:

- data/attendance_DD-MM-YYYY.csv

## 6. Generate Faculty Summary

```bash
python generate_summary.py
```

Expected output:

- data/faculty_submission.csv

## 7. Optional Utilities

- Vault management:

```bash
python vault_manager.py
```

- Diagnostic view:

```bash
python diagnostic_view.py
```

## 8. Attendance Video Source

Use this provided Drive folder for attendance video input files:

- https://drive.google.com/drive/folders/1jJ__zesfwOiY8FWt1CGb0mFrPOCCbkwc?usp=drive_link

## 9. Common Quick Fixes

- Error: Vault not found
  Run enrollmentupd.py first.
- Camera not opening
  Close other apps using the camera and retry.
- Too many Unknown detections
  Re-enroll students in better lighting and frontal pose.
