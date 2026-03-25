import cv2
import os
import time
from ultralytics import YOLO


person_detector = YOLO('yolov8n.pt')

# helper function to set vid. capture params
def capture_setter(vid_path):
    cap = cv2.VideoCapture(vid_path)
    frame_width = 1280
    frame_height = 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    return cap

# helper function for saving trackets as folders
def save_tracklets(frame, track, base_dir = 'tracklets', save_every = 20, blur_thresh = 100):
    if not track.is_confirmed():
        return # save only confirmed tracks to avoid noise
    track_id = track.track_id
    if not hasattr(track, 'save_counter'): # counts how many times a track has been saved 
        track.save_counter = 0
    track.save_counter += 1
    if track.save_counter % save_every != 0:
        return # 1 in 3
    
    # get bbox coords
    l, t, r, b = track.to_ltrb()
    l, t, r, b = map(int, [l, t, r, b])

    # frame boundary check
    h_frame, w_frame = frame.shape[:2]
    l = max(0, l)
    t = max(0, t)
    r = min(w_frame, r)
    b = min(h_frame, b)

    # size sanity check
    if (r-1) < 60 or (b-t) < 120:
        return
    crop = frame[t:b, l:r]
    if crop.size == 0:
        return
    
    # blur detection by laplacian invariance
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    if blur_score < blur_thresh:
        return
    
    # create a folder for the person and save the bbox crops
    person_dir = os.path.join(base_dir, f"person_{track_id}")
    os.makedirs(person_dir, exist_ok=True)
    # save image
    img_name = f"{track_id}_{int(time.time()*1000)}.jpg"
    cv2.imwrite(os.path.join(person_dir, img_name), crop)

def get_person_detections(frame):
    
    # Run YOLO inference
    results = person_detector.predict(
        frame,
        imgsz=640,
        conf=0.45,          # confidence threshold
        iou=0.7,            # NMS IoU threshold
        classes=[0],        # only person class
        save=False,         # do not save/visualize internally
        verbose=False
    )

    detections = []
    r = results[0]  # first batch result

    if r.boxes is not None:
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        for box, conf in zip(xyxy, confs):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            
            # DeepSORT expects ([x, y, w, h], confidence)
            detections.append(([x1, y1, w, h], float(conf)))

    return detections