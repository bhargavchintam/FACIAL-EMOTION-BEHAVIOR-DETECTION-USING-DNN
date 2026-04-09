# Facial Emotion & Behavior Detection using DNN

Comprehensive collection of deep-learning demos for real-time facial analysis: emotion recognition, drowsiness, mask usage, age and gender estimation, and social-distancing monitoring. Each module is self-contained with pretrained weights and runnable scripts.

## Repository Layout
```
Final Codes/
  Drowsiness Detection/         # Eye-aspect-ratio drowsiness alarm (dlib, OpenCV)
  Emotion Detection/            # FER2013-based emotion classifier with live webcam demo
  Face Mask Detection/          # SSD-based mask detector with multiple framework runtimes
  Gender and Age Detection/     # Caffe age/gender classifiers with sample images
  Social Distancing AI/         # Perspective-calibrated distance checker with homography
  Social Distancing Detection/  # YOLOv3 social-distancing monitor with alert hooks
Model Diagrams Pics/            # Presentation and architecture visuals
Project Images and Files/       # Sample outputs and supporting images
Paper Acceptance Mail.jpg, Patent Application Form.png, PPT format.mp4, SECURITY.md, etc.
```

## Architecture at a Glance
- **Input capture:** Webcam or video files feed each detection task.
- **Detection backbones:** dlib facial landmarks for EAR; CNNs for emotion, mask, and age/gender; YOLOv3 for person localization; SSD-lite for mask detection.
- **Post-processing:** Thresholds/ratios for drowsiness, probability bars for emotions, distance calculations for social-distancing, and alarm hooks for alerts.
- **Outputs:** On-frame overlays, probability charts, and optional audio alerts.

![Social distancing pipeline](Final%20Codes/Social%20Distancing%20AI/images/block_diagram.png)

## Quickstart
1. Install Python 3.7+ and create a virtual environment.
2. Move into the module you want to run and install its dependencies (requirements differ by module).
3. Use the commands below to launch a demo.

### Drowsiness Detection
```bash
cd "Final Codes/Drowsiness Detection"
pip install imutils dlib playsound opencv-python scipy
python detect_drowsiness.py --shape-predictor model/shape_predictor_68_face_landmarks.dat --alarm sounds/alarm.wav
```

### Emotion Detection
```bash
cd "Final Codes/Emotion Detection"
pip install -r requirements.txt
python real_time_video.py         # webcam probabilities overlay
# Optional: python train_emotion_classifier.py to retrain on FER2013 (place CSV in fer2013/fer2013/)
```

### Face Mask Detection
```bash
cd "Final Codes/Face Mask Detection"
python pytorch_infer.py --img-path /path/to/image.jpg          # PyTorch example
python tensorflow_infer.py --img-path /path/to/image.jpg       # TensorFlow example
```
*Notes: Models for TensorFlow/Keras/MXNet/Caffe are included under `img/` and related folders. Use `--img-mode 0 --video-path 0` to run on webcam.*

### Gender & Age Detection
```bash
cd "Final Codes/Gender and Age Detection"
pip install opencv-python argparse
python detect.py --image girl1.jpg   # classify a still image in this folder
python detect.py                     # run on webcam
```
Pretrained Caffe models (`age_net.caffemodel`, `gender_net.caffemodel`) and face detector weights are already included.

### Social Distancing Detection (YOLOv3)
```bash
cd "Final Codes/Social Distancing Detection"
pip install -r requirements.txt
python Run.py -i mylib/videos/test.mp4      # analyze a video file
python Run.py                               # default webcam (set url in mylib/config.py for IP cams)
```
*Notes: YOLOv3 weights/config are preloaded under `yolo/`. Enable GPU by setting `USE_GPU = True` in `mylib/config.py`. Email alert configuration lives in `mylib/mailer.py`.*

### Social Distancing AI (Perspective Calibration)
```bash
cd "Final Codes/Social Distancing AI"
pip install -r requirements.txt
python main.py --videopath "vid_short.mp4"
```
On launch, click six calibration points (four ROI corners + two reference points six feet apart) before processing begins.

## Sample Outputs
![Emotion probabilities](Project%20Images%20and%20Files/Probabilities.png)
![Age and gender prediction](Project%20Images%20and%20Files/Detecting%20age%20and%20gender3.png)
![Social distancing calibration](Final%20Codes/Social%20Distancing%20AI/images/cover.png)

## Additional Notes
- Large supporting docs, papers, and presentations live under the `Final IEEE Paper`, `Final Scopus Paper`, and related folders.
- Security guidelines are in `SECURITY.md`.
- If you retrain models, keep artifacts in their respective module folders to match the run scripts.

