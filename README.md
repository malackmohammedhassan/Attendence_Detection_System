Smart Attendance System using Face Recognition
Project Overview
This project presents a Smart Attendance System that automates student attendance using face detection and face recognition.
The system is designed for classroom and lab environments, where student presence is verified in real time and attendance is stored digitally for reporting and faculty use.

The solution combines modern computer vision and deep learning methods to improve reliability, reduce manual effort, and minimize proxy attendance.

Objective
Automate attendance marking through facial recognition
Improve speed and accuracy compared to manual methods
Maintain secure biometric records for enrolled students
Generate clean attendance reports for faculty submission
Key Features
Real-time face detection and recognition
Student enrollment with multi-angle face capture
Live attendance logging with date and time
Session-wise attendance CSV generation
Faculty-friendly attendance summary export
Support for GPU acceleration (where available)
Clean and practical interface for classroom operation
Methodology
Enrollment Phase
Students are enrolled by capturing facial samples from multiple angles. Embeddings are generated and stored in a secure biometric vault.

Recognition Phase
During attendance, faces are detected from live camera or video input. New embeddings are compared against enrolled templates using cosine similarity and threshold-based matching.

Attendance Generation
Recognized students are marked present, duplicates are avoided, and attendance data is exported in structured CSV format.

Tech Stack
Python
OpenCV
Ultralytics YOLO (face detection)
InsightFace ArcFace (face recognition embeddings)
NumPy and Pandas
PyTorch (GPU acceleration support)
Academic Relevance
This project applies soft computing and AI-based pattern recognition techniques to solve a real institutional problem.
It demonstrates practical integration of machine learning, biometric security, and automation in educational systems.

Expected Outcomes
Reduced attendance time in classrooms
Better authenticity of attendance records
Scalable architecture for wider campus deployment
Data-driven reporting for academic administration
Limitations
Recognition quality may reduce under poor lighting or extreme pose variation
Requires proper enrollment quality for best performance
Camera quality and placement influence performance
Future Scope
Web dashboard integration for department-level analytics
Liveness detection to prevent spoofing attacks
Mobile application support for faculty access
Cloud deployment for centralized attendance management
How to Run
Install project dependencies.
Enroll students to create the biometric vault.
Start live attendance recognition.
Export attendance report and faculty summary.
Acknowledgement
This project was developed as part of academic coursework in Soft Computing.
Guidance provided where required.
I sincerely acknowledge the support of my faculty mentor, department, and institution for enabling this implementation.

Declaration
This repository is submitted for academic and demonstration purposes.
All implementation, testing, and documentation were prepared as part of project learning and evaluation.
