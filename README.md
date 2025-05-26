# Drowsiness-detection-system
Real-time driver drowsiness detection using CNN (MobileNetV2), OpenCV, and alert system
## 📌 Project Description

This project uses a custom-trained MobileNetV2 CNN model to detect whether a driver's eyes are open or closed using webcam input. It integrates face and eye detection using OpenCV, and raises alerts with audio and visual cues when drowsiness is detected.

The system monitors eye activity in real time, and if eyes remain closed for more than 500 milliseconds, it plays an alert sound and flashes a red overlay on the screen—instantly waking the driver and preventing potential accidents. The entire detection pipeline works efficiently on low-end systems, making it a practical safety solution.

---

## ⚠️ Problems Faced & How They Were Solved

### 1. **Real-time Performance with Model Lag**
- **Problem**: Initial model prediction was slow and caused webcam lag.
- **Solution**: Used `MobileNetV2` with reduced input size (96x96) for fast inference.

### 2. **False Alarms with Haar Cascade Eye Detection**
- **Problem**: Haar cascades misidentified areas as eyes, especially with glasses or side angles.
- **Solution**: Integrated prediction probability threshold (`< 0.3 = closed`) and filtered multiple detections by region and size. Ensured eyes were correctly captured before feeding into model.

### 3. **Alert Sound Not Playing or Crashing**
- **Problem**: `playsound` module failed due to MCI device issues.
- **Solution**: Switched to multithreaded sound playing using daemon threads. Added try-except to avoid crash if sound fails.

### 4. **No Face Detection for Better Context**
- **Problem**: Only eyes were being detected; face context was missing.
- **Solution**: Added OpenCV Haar Cascade face detection alongside eye detection to improve stability and reduce false positives.

### 5. **Unreliable Alerts Due to Frame Delay**
- **Problem**: Delay in processing frames made detection unstable.
- **Solution**: Tuned the 0.3-second threshold for eye closure and avoided processing multiple frames in tight loops.

---

## 🧠 Best Practices for Optimal Model Performance

- ✅ Keep the camera **at eye level** or **slightly above** for best visibility of the eyes.
- ✅ Ensure **adequate lighting** and **avoid occlusion** (e.g., hands/glasses).
- ✅ Use a **clean webcam lens** to avoid blurring or misclassification.

---

## 🔧 Tech Stack

- TensorFlow / Keras (MobileNetV2 model)
- OpenCV (Face and Eye Detection)
- NumPy
- playsound (audio alerts)
- threading (non-blocking audio execution)

DATASET- https://drive.google.com/drive/folders/1ixP6C7wiLs8lO71T5QGUIJSBKKX56u7B?usp=drive_link

PS:I have used Chatgpt to Create the Read me file to make it more Readable
