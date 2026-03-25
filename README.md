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
