# Smart Attendance System Using Face Recognition

## Abstract

This project implements an AI-powered Smart Attendance System that automates student attendance using real-time face detection and recognition. The system combines YOLO-based face detection with ArcFace embeddings to identify enrolled students and record attendance with date, time, and status. It is designed for classroom and laboratory environments to reduce manual effort, improve record authenticity, and provide faculty-ready attendance reports.

## Problem Statement

Traditional attendance methods are time-consuming, error-prone, and vulnerable to proxy attendance. This project addresses these limitations by using biometric verification through facial recognition.

## Objectives

- Automate student attendance with minimal manual intervention.
- Improve reliability and reduce proxy attendance.
- Maintain structured digital attendance logs for academic reporting.
- Provide a practical, deployable system for real institutional use.

## System Highlights

- Real-time attendance from webcam/live feed.
- Student enrollment with multi-angle face capture.
- Secure biometric vault for face embeddings.
- Threshold- and margin-based matching for robust identification.
- Duplicate attendance prevention in a single session.
- CSV attendance export and faculty summary generation.
- GPU-supported execution when CUDA is available.

## Methodology

### 1. Enrollment Phase

Student identity data (name and registration number) is captured along with multiple facial views. The system extracts face embeddings and stores them in a biometric vault.

### 2. Recognition Phase

During live attendance, faces are detected frame-by-frame and converted into embeddings. Similarity is computed against enrolled templates using cosine similarity.

### 3. Decision Logic

A student is marked present only when the confidence threshold and inter-class margin checks are satisfied. This reduces false matches.

### 4. Attendance Logging

Recognized students are logged with date, time, slot, and status. Session reports are generated as CSV files for documentation and submission.

## Tech Stack

- Python
- OpenCV
- Ultralytics YOLO (face detection)
- InsightFace ArcFace (face recognition)
- NumPy, Pandas
- PyTorch

## Project Structure (Key Files)

- main.py: live attendance interface and logging pipeline
- enrollmentupd.py: student enrollment and biometric vault creation
- process_videoupd.py: attendance processing from video with reporting
- generate_summary.py: faculty submission summary generation
- data/: attendance CSV outputs and vault data
- docs/: project documentation and technical notes

## How to Run Both Systems Separately

### A) Attendance System (Desktop Module)

Location:

- attendance*system_soft_computing*

Quick run order:

1. Install dependencies:
   - `pip install -r requirements2.txt`
2. Enroll students:
   - `python enrollmentupd.py`
3. Run live attendance:
   - `python main.py`
4. Run CCTV/video attendance (optional):
   - `python process_videoupd.py --source "C:\\path\\to\\video.mp4"`
5. Generate faculty summary:
   - `python generate_summary.py`

Detailed run instructions file:

- [RUN_ATTENDANCE_SYSTEM.md](RUN_ATTENDANCE_SYSTEM.md)

Attendance video source (Google Drive):

- https://drive.google.com/drive/folders/1jJ__zesfwOiY8FWt1CGb0mFrPOCCbkwc?usp=drive_link

### B) Face WebUI Platform (Web Module)

Location:

- face-webui-platform

Recommended start (Docker):

1. Open terminal in `face-webui-platform`
2. Run:
   - `docker compose up --build`
3. Access:
   - Frontend: http://localhost:5173
   - Backend API: http://localhost:8000

Detailed run instructions file:

- [RUN_FACE_WEBUI_PLATFORM.md](../../face-webui-platform/Docs/RUN_FACE_WEBUI_PLATFORM.md)

Related combined implementation guide:

- [IMPLEMENTATION_AND_FUNCTIONALITY.md](../../Docs/IMPLEMENTATION_AND_FUNCTIONALITY.md)

## Academic Contribution

This work demonstrates practical application of soft computing and computer vision in educational automation. It integrates machine learning-based pattern recognition with an operational workflow suitable for institutional deployment.

## Results and Impact

- Reduced attendance handling time.
- Improved integrity of attendance records.
- Structured digital reporting for faculty and administration.
- Scalable foundation for campus-wide smart attendance systems.

## Limitations

- Performance can degrade under poor lighting or heavy occlusion.
- Enrollment quality directly affects recognition quality.
- Camera placement and resolution influence accuracy.

## Future Scope

- Liveness detection to prevent spoofing.
- Web dashboard for multi-class analytics.
- Cloud synchronization for centralized records.
- Mobile faculty access for attendance monitoring.

## Acknowledgement

This project was developed as part of academic coursework in Soft Computing. Guidance was provided where required. The support of faculty mentors and the department is gratefully acknowledged.

## Declaration

This repository is prepared for academic evaluation and demonstration purposes. The implementation and documentation reflect project-based learning and applied research in intelligent systems.
