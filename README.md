# 📹🤖 SmartLogger-YOLOv8n-DeepSORT-OSNet1.0x-Based-Unique-Visitor-Logging-System
This repo implements a 3-stage unique visitor log generation pipeline that takes CCTV footage over a timeline and generates a list of all unique persons who have appeared in the video. YOLOv8n and DeepSORT are used to generate tracklets &amp; OSNet-1.0x is used to compute multipose avg. embeddings which are matched to obtain unique individuals.

# Demo 👇
<video src="demo.mp4" controls width="640"></video>
[[Link to Demo]](https://youtu.be/jeZw4MNxz0s "Click to watch")

# Overview of the pipeline
![Alt text](Full_Pipeline.png)

## 🚀 Features

* **YOLOv8n**: Accurate **frame-by-frame** person detection.
* **DeepSORT**: Tracks persons frame-by-frame and extracts stable **tracklets** containing person appearences in **multiple poses**.
* **OSNet-1.0x**: Considers **each image** in the tracklets folder per person and computes a **512-d multi-pose avg. embedding**.
---

## 📂 Project Structure

```bash
.
├── yolov8n.pt              # YOLOv8n with COCO weights downloaded from Ultralytics.
├── utils.py                # Utility functions for aggregating detections & generating tracklets
├── yolov8n_inference.ipynb          # Runs the entire log generation pipeline: Detection+Tracking --> Tracklets generation --> Embedding & Similarity matching --> Log generation.
├── tracklets/           # Stores tracklets upon running the pipeline.
├── videos/              # [To be downloaded] A sample video for inference.
├── torchreid/           # [To be downloaded] package holding the dependencies of OSNet embedding generation.
├── unique_people/       # Stores a log of unique people which can be viewed after video inference ends.
├── reid_model/                   # [To be downloaded] Stores the OSNet_1.0x MSMT17 checkpoint used for embedding generation.
       ├── osnet_x1_0_msmt17.pth
├── requirements.txt      # Python dependencies.
├── config.yaml         # parameters concerning video display.
├── osnet_log_generator.py  # Utility functions for generating the unique log via. similarity matching.
├── tracker.py            # The class definition of the DeepSORT tracker injected with optimal parameters for seamless frame-by-frame tracking.
