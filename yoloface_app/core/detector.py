"""
Face Detection Worker - Background thread for processing
"""

import os
import sys
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, QMutex
import time

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils import get_outputs_names, post_process, IMG_WIDTH, IMG_HEIGHT, CONF_THRESHOLD, NMS_THRESHOLD


class FaceTrackerCore:
    """Core face tracking logic (adapted from yoloface_tracker.py)"""
    
    def __init__(self, tolerance=0.6):
        self.next_id = 0
        self.tolerance = tolerance
        self.face_count = 0
        self.face_samples = {}
        self.face_best = {}
        self.face_last_seen = {}
        self.face_positions = {}
        self.min_samples_to_collect = 15
        self.timeout_frames = 30
    
    def calculate_iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def find_matching_face_by_position(self, box, iou_threshold=0.3):
        best_match_id = None
        best_iou = iou_threshold
        
        for face_id, last_pos in self.face_positions.items():
            iou = self.calculate_iou(box, last_pos)
            if iou > best_iou:
                best_iou = iou
                best_match_id = face_id
        
        return best_match_id
    
    def calculate_image_quality(self, face_region):
        if face_region is None or face_region.size == 0:
            return 0
        
        height, width = face_region.shape[:2]
        size_score = min(height * width / 10000, 1.0)
        
        try:
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 100, 1.0)
            quality_score = (size_score * 0.4) + (sharpness_score * 0.6)
        except:
            quality_score = size_score
        
        return quality_score
    
    def add_face_sample(self, face_id, face_img, frame_num):
        if face_id not in self.face_samples:
            self.face_samples[face_id] = []
        
        quality = self.calculate_image_quality(face_img)
        self.face_samples[face_id].append((face_img.copy(), quality, frame_num))
        
        return quality
    
    def get_best_face_image(self, face_id):
        if face_id not in self.face_samples or len(self.face_samples[face_id]) == 0:
            return None
        
        best_img, best_score, best_frame = max(self.face_samples[face_id], key=lambda x: x[1])
        return best_img, best_score, best_frame


class DetectionWorker(QThread):
    """Worker thread for face detection"""
    
    # Signals
    frame_ready = pyqtSignal(np.ndarray)  # Processed frame with rectangles
    face_detected = pyqtSignal(int, np.ndarray, tuple)  # face_id, face_image, box
    new_face = pyqtSignal(int, np.ndarray)  # face_id, face_image
    sample_added = pyqtSignal(int, int)  # face_id, sample_count
    face_saved = pyqtSignal(int, str)  # face_id, file_path
    log_message = pyqtSignal(str, str)  # message, level
    progress_update = pyqtSignal(float)  # progress percent
    stats_update = pyqtSignal(int, int, float)  # total_faces, current_frame, fps
    processing_complete = pyqtSignal(str)  # summary message
    error = pyqtSignal(str)
    
    def __init__(self, source_type, source_path, settings, parent=None):
        super().__init__(parent)
        self.mutex = QMutex()
        self._running = False
        self._paused = False
        
        # Source
        self.source_type = source_type
        self.source_path = source_path
        
        # Settings
        self.tolerance = settings.get('tolerance', 0.6)
        self.skip_frames = settings.get('skip_frames', 10)
        self.sample_timeout = settings.get('sample_timeout', 3)
        self.save_output = settings.get('save_output', True)
        self.save_faces = settings.get('save_faces', True)
        self.show_boxes = settings.get('show_boxes', True)
        self.fast_mode = settings.get('fast_mode', True)
        self.output_dir = 'outputs'
        
        # Model
        self.net = None
        
    def set_paused(self, paused):
        """Set pause state"""
        self.mutex.lock()
        self._paused = paused
        self.mutex.unlock()
        
    def load_model(self):
        """Load the YOLO model"""
        try:
            # Get absolute paths
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            cfg_path = os.path.join(base_dir, 'cfg', 'yolov3-face.cfg')
            weights_path = os.path.join(base_dir, 'model-weights', 'yolov3-wider_16000.weights')
            
            # Print config info to terminal
            print("[i] The config file: ", cfg_path)
            print("[i] The weights of model file: ", weights_path)
            
            self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.log_message.emit("YOLO model loaded successfully", "success")
            return True
        except Exception as e:
            self.error.emit(f"Failed to load model: {str(e)}")
            return False
    
    def stop(self):
        """Stop processing"""
        self.mutex.lock()
        self._running = False
        self.mutex.unlock()
    
    def run(self):
        """Main processing loop"""
        self._running = True
        self._paused = False
        
        # Print settings to terminal
        print("###########################################################")
        print(f"[i] Source type: {self.source_type}")
        print(f"[i] Source path: {self.source_path}")
        print(f"[i] Tolerance: {self.tolerance}")
        print(f"[i] Save faces: {self.save_faces}")
        print(f"[i] Save output: {self.save_output}")
        print(f"[i] Fast mode: {self.fast_mode}")
        print(f"[i] Skip frames: {self.skip_frames}")
        print("###########################################################\n")
        
        # Load model if not loaded
        if self.net is None:
            if not self.load_model():
                return
        
        # Initialize face tracker
        face_tracker = FaceTrackerCore(tolerance=self.tolerance)
        face_tracker.timeout_frames = int(self.sample_timeout * 30)  # Assuming ~30 fps
        
        # Setup capture
        cap = None
        total_frames = 0
        fps_timer = time.time()
        fps_counter = 0
        current_fps = 0.0
        
        if self.source_type == 'webcam':
            cap = cv2.VideoCapture(0)
            total_frames = -1  # Unknown for webcam
        elif self.source_type == 'video':
            cap = cv2.VideoCapture(self.source_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        elif self.source_type == 'image':
            cap = cv2.VideoCapture(self.source_path)
            total_frames = 1
        
        if cap is None or not cap.isOpened():
            self.error.emit("Failed to open source")
            return
        
        # Setup output directory
        faces_dir = os.path.join(self.output_dir, 'distinct_faces')
        if os.path.exists(faces_dir):
            print(f"==> Skipping create the {self.output_dir}/ directory...")
        else:
            os.makedirs(faces_dir, exist_ok=True)
        print(f"==> Faces directory: {faces_dir}")
        
        # Video writer setup
        video_writer = None
        if self.save_output and self.source_type in ['video', 'webcam']:
            output_path = os.path.join(self.output_dir, 'output_detection.avi')
            fps = cap.get(cv2.CAP_PROP_FPS) if self.source_type == 'video' else 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                fps, (width, height)
            )
        
        start_time = time.time()
        frame_count = 0
        
        self.log_message.emit("Starting face detection...", "info")
        
        while self._running:
            # Check if paused
            while self._paused and self._running:
                time.sleep(0.1)
            
            if not self._running:
                break
            
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            fps_counter += 1
            
            # Calculate FPS every second
            if time.time() - fps_timer >= 1.0:
                current_fps = fps_counter
                fps_counter = 0
                fps_timer = time.time()
            
            # Skip frames for faster processing
            if self.source_type != 'image' and self.skip_frames > 1:
                if frame_count % self.skip_frames != 0:
                    continue
            
            # Keep clean frame for saving faces
            clean_frame = frame.copy()
            
            # Run YOLO detection
            blob = cv2.dnn.blobFromImage(frame, 1/255, (IMG_WIDTH, IMG_HEIGHT),
                                         [0, 0, 0], 1, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(get_outputs_names(self.net))
            
            # Post-process detections
            faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
            
            # Log frame info periodically
            if frame_count % 50 == 0 or self.source_type == 'image':
                print(f"[Frame {frame_count}] # detected faces: {len(faces)}")
                self.log_message.emit(f"[Frame {frame_count}] Detected {len(faces)} faces", "info")
            
            detected_faces_in_frame = set()
            
            # Process each detected face
            for face_coords in faces:
                left, top, width, height = int(face_coords[0]), int(face_coords[1]), int(face_coords[2]), int(face_coords[3])
                x1, y1, x2, y2 = left, top, left + width, top + height
                
                current_box = (x1, y1, x2, y2)
                
                # Match with existing face or create new
                face_id = face_tracker.find_matching_face_by_position(current_box)
                is_new_face = face_id is None
                
                if face_id is None:
                    face_id = face_tracker.next_id
                    face_tracker.next_id += 1
                    face_tracker.face_count += 1
                
                # Update tracking
                face_tracker.face_positions[face_id] = current_box
                face_tracker.face_last_seen[face_id] = frame_count
                detected_faces_in_frame.add(face_id)
                
                # Extract face region
                margin = 20
                face_x1 = max(0, x1 - margin)
                face_y1 = max(0, y1 - margin)
                face_x2 = min(frame.shape[1], x2 + margin)
                face_y2 = min(frame.shape[0], y2 + margin)
                
                face_img = clean_frame[face_y1:face_y2, face_x1:face_x2].copy()
                
                if face_img.size > 0:
                    is_first_sample = face_id not in face_tracker.face_samples or len(face_tracker.face_samples.get(face_id, [])) == 0
                    
                    quality = face_tracker.add_face_sample(face_id, face_img, frame_count)
                    samples_count = len(face_tracker.face_samples.get(face_id, []))
                    
                    # Emit new face signal and save initial image immediately
                    if is_new_face:
                        print(f"  [NEW] Face ID #{face_id} detected at frame {frame_count}")
                        print(f"  [SAMPLE] Face ID #{face_id} - Sample {samples_count} (Quality: {quality:.2f})")
                        self.log_message.emit(f"[NEW] Face ID #{face_id} detected", "success")
                        self.new_face.emit(face_id, face_img)
                        
                        # Save initial face image immediately to disk
                        if self.save_faces:
                            face_path = os.path.join(faces_dir, f'face_id_{face_id:03d}.jpg')
                            cv2.imwrite(face_path, face_img)
                            face_tracker.face_best[face_id] = (face_path, quality)
                            self.face_saved.emit(face_id, face_path)
                    elif samples_count <= 5 or samples_count % 10 == 0:
                        # Print sample info for first few samples or every 10th
                        print(f"  [SAMPLE] Face ID #{face_id} - Sample {samples_count} (Quality: {quality:.2f})")
                    
                    # Emit sample added signal
                    self.sample_added.emit(face_id, samples_count)
                    
                    # Emit face detected for display updates
                    self.face_detected.emit(face_id, face_img, current_box)
                
                # Draw bounding box if enabled
                if self.show_boxes:
                    color = (0, 255, 0)  # Green
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'ID: {face_id}', (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Check for timeout faces - update with best quality image
            faces_to_update = []
            for fid in list(face_tracker.face_samples.keys()):
                if fid not in detected_faces_in_frame:
                    frames_since = frame_count - face_tracker.face_last_seen.get(fid, frame_count)
                    samples = len(face_tracker.face_samples[fid])
                    
                    if frames_since >= face_tracker.timeout_frames and samples > 0:
                        faces_to_update.append(fid)
            
            # Update timeout faces with best quality image
            for fid in faces_to_update:
                if self.save_faces:
                    result = face_tracker.get_best_face_image(fid)
                    if result:
                        best_img, best_score, _ = result
                        # Use same path as initial save (overwrite with best)
                        face_path = os.path.join(faces_dir, f'face_id_{fid:03d}.jpg')
                        cv2.imwrite(face_path, best_img)
                        face_tracker.face_best[fid] = (face_path, best_score)
                        samples = len(face_tracker.face_samples.get(fid, []))
                        self.log_message.emit(f"[UPDATED] Face #{fid} with best quality ({best_score:.2f})", "success")
                        # Clear samples after updating with best
                        face_tracker.face_samples[fid] = []
            
            # Emit frame for display
            self.frame_ready.emit(frame)
            
            # Update stats
            self.stats_update.emit(face_tracker.face_count, frame_count, current_fps)
            
            # Update progress
            if total_frames > 0:
                progress = (frame_count / total_frames) * 100
                self.progress_update.emit(progress)
            
            # Save to video writer
            if video_writer:
                video_writer.write(frame)
            
            # For image, just process once
            if self.source_type == 'image':
                # Animate progress bar over ~2 seconds (emit progress in steps)
                for pct in [20, 40, 60, 80, 100]:
                    self.progress_update.emit(float(pct))
                    time.sleep(0.35)
                
                # Save image with detections
                if self.save_output:
                    output_path = os.path.join(self.output_dir, 'output_detection.jpg')
                    cv2.imwrite(output_path, frame)
                
                # Emit the annotated frame (with boxes) so UI shows detections
                self.frame_ready.emit(frame)
                
                # Save all detected faces for image
                if self.save_faces:
                    for fid in face_tracker.face_samples.keys():
                        if fid not in face_tracker.face_best:
                            result = face_tracker.get_best_face_image(fid)
                            if result:
                                best_img, best_score, _ = result
                                face_path = os.path.join(faces_dir, f'face_id_{fid:03d}.jpg')
                                cv2.imwrite(face_path, best_img)
                                face_tracker.face_best[fid] = (face_path, best_score)
                                self.face_saved.emit(fid, face_path)
                break
            
            # Small delay for smoother processing
            time.sleep(0.01)
        
        # Update remaining faces at end with best quality
        print("\n[i] Updating remaining faces with best quality...")
        self.log_message.emit("Updating remaining faces with best quality...", "info")
        if self.save_faces:
            for fid in list(face_tracker.face_samples.keys()):
                samples = len(face_tracker.face_samples.get(fid, []))
                if samples > 0:
                    result = face_tracker.get_best_face_image(fid)
                    if result:
                        best_img, best_score, _ = result
                        # Use same path as initial save (overwrite with best)
                        face_path = os.path.join(faces_dir, f'face_id_{fid:03d}.jpg')
                        cv2.imwrite(face_path, best_img)
                        face_tracker.face_best[fid] = (face_path, best_score)
                        print(f"  [UPDATED] Face #{fid} with best quality ({best_score:.2f})")
                        self.log_message.emit(f"[UPDATED] Face #{fid} (Quality: {best_score:.2f})", "success")
        
        # Cleanup
        elapsed_time = time.time() - start_time
        
        if cap:
            cap.release()
        if video_writer:
            video_writer.release()
        
        print(f"\n==> Total distinct faces detected: {face_tracker.face_count}")
        print(f"Total execution time: {elapsed_time:.2f} seconds")
        summary = f"Detected {face_tracker.face_count} distinct faces in {elapsed_time:.2f}s"
        self.log_message.emit(summary, "success")
        self.processing_complete.emit(summary)
