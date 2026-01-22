import argparse
import sys
import os
import numpy as np
import cv2
import pickle
from pathlib import Path
from utils import *
import face_recognition
import mss
import time
import mss.tools
    
##########
parser = argparse.ArgumentParser()
parser.add_argument('--model-cfg', type=str, default='./cfg/yolov3-face.cfg',
                    help='path to config file')
parser.add_argument('--model-weights', type=str,
                    default='./model-weights/yolov3-wider_16000.weights',
                    help='path to weights of model')
parser.add_argument('--image', type=str, default='',
                    help='path to image file')
parser.add_argument('--video', type=str, default='',
                    help='path to video file')
parser.add_argument('--src', type=int, default=0,
                    help='source of the camera')
parser.add_argument('--output-dir', type=str, default='outputs/',
                    help='path to the output directory')
parser.add_argument('--tolerance', type=float, default=0.6,
                    help='face recognition tolerance (lower = stricter matching)')
parser.add_argument('--save-faces', action='store_true',
                    help='save detected faces to disk')
parser.add_argument('--save-video', action='store_true',
                    help='save output video with detections')
parser.add_argument('--display', action='store_true',
                    help='display video while processing')
parser.add_argument('--screen', action='store_true',
                    help='capture screen and detect faces')
parser.add_argument('--monitor', type=int, default=1,
                    help='monitor number for screen capture (default: 1 = primary)')
parser.add_argument('--fast-mode', action='store_true',
                    help='fast mode: save all faces without encoding (no duplicate detection)')
parser.add_argument('--skip-frames', type=int, default=10,
                    help='in fast mode, save faces every N frames (default: 10)')
args = parser.parse_args()

#####################################################################
# print the arguments
print('----- info -----')
print('[i] The config file: ', args.model_cfg)
print('[i] The weights of model file: ', args.model_weights)
print('[i] Path to image file: ', args.image)
print('[i] Path to video file: ', args.video)
print('[i] Screen capture: ', args.screen)
if args.screen:
    print('[i] Monitor: ', args.monitor)
print('[i] Tolerance: ', args.tolerance)
print('[i] Save faces: ', args.save_faces)
print('[i] Save video: ', args.save_video)
print('[i] Display video: ', args.display)
print('[i] Fast mode: ', args.fast_mode)
if args.fast_mode:
    print('[i] Skip frames: ', args.skip_frames)
print('###########################################################\n')

# check outputs directory
if not os.path.exists(args.output_dir):
    print('==> Creating the {} directory...'.format(args.output_dir))
    os.makedirs(args.output_dir)
else:
    print('==> Skipping create the {} directory...'.format(args.output_dir))

# Create faces output directory
faces_dir = os.path.join(args.output_dir, 'distinct_faces')
if not os.path.exists(faces_dir):
    os.makedirs(faces_dir)
    print(f'==> Created faces directory: {faces_dir}')

# Load the network
net = cv2.dnn.readNetFromDarknet(args.model_cfg, args.model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


class FaceTracker:
    def __init__(self, tolerance=0.6):
        self.known_face_encodings = []
        self.known_face_ids = []
        self.next_id = 0
        self.tolerance = tolerance
        self.face_count = 0
        self.face_samples = {}  # Store multiple images per face: {face_id: [(image, score, frame_num), ...]}
        self.face_best = {}  # Store best image per face: {face_id: (image_path, score)}
        self.face_last_seen = {}  # Track last frame each face was seen: {face_id: frame_num}
        self.face_positions = {}  # Track last position of each face: {face_id: (x1, y1, x2, y2)}
        self.min_samples_to_collect = 15  # Collect 15 frames before selecting best
        self.timeout_frames = 30  # Save best after 30 frames without detection
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def find_matching_face_by_position(self, box, iou_threshold=0.3):
        """Find a tracked face that matches this position"""
        best_match_id = None
        best_iou = iou_threshold
        
        for face_id, last_pos in self.face_positions.items():
            iou = self.calculate_iou(box, last_pos)
            if iou > best_iou:
                best_iou = iou
                best_match_id = face_id
        
        return best_match_id
        
    def calculate_image_quality(self, face_region):
        """Calculate quality score for a face region (higher is better)"""
        if face_region is None or face_region.size == 0:
            return 0
        
        # Score based on face region size (larger = better)
        height, width = face_region.shape[:2]
        size_score = min(height * width / 10000, 1.0)  # Normalize
        
        # Score based on image brightness/contrast (Laplacian variance for sharpness)
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 100, 1.0)  # Normalize
        
        # Combined score
        quality_score = (size_score * 0.4) + (sharpness_score * 0.6)
        return quality_score
        
    def add_face_sample(self, face_id, face_img, frame_num):
        """Add a sample image for a face"""
        if face_id not in self.face_samples:
            self.face_samples[face_id] = []
        
        quality = self.calculate_image_quality(face_img)
        self.face_samples[face_id].append((face_img.copy(), quality, frame_num))
        
        return quality
    
    def get_best_face_image(self, face_id):
        """Get the best quality image for a face"""
        if face_id not in self.face_samples or len(self.face_samples[face_id]) == 0:
            return None
        
        # Find the image with highest quality score
        best_img, best_score, best_frame = max(self.face_samples[face_id], key=lambda x: x[1])
        return best_img, best_score, best_frame
        
    def get_face_encoding(self, frame, face_box):
        """Extract face encoding from a face region"""
        try:
            # Convert BGR to RGB for face_recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get face location from bbox
            x1, y1, x2, y2 = face_box
            
            # Use face_recognition to detect faces in full frame first
            face_locations = face_recognition.face_locations(rgb_frame, model='hog')
            
            if len(face_locations) == 0:
                return None
            
            # Find the face location that best matches our YOLO bbox
            best_encoding = None
            best_overlap = 0
            
            encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            for i, (top, right, bottom, left) in enumerate(face_locations):
                # Calculate overlap with YOLO bbox
                overlap_x1 = max(x1, left)
                overlap_y1 = max(y1, top)
                overlap_x2 = min(x2, right)
                overlap_y2 = min(y2, bottom)
                
                if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                    overlap = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_encoding = encodings[i]
            
            return best_encoding
        except Exception as e:
            print(f"[!] Error getting face encoding: {e}")
            return None
    
    def is_new_face(self, encoding):
        """Check if this is a new distinct face"""
        if len(self.known_face_encodings) == 0:
            return True
        
        # Compare with all known faces
        distances = face_recognition.face_distance(self.known_face_encodings, encoding)
        min_distance = np.min(distances)
        
        return min_distance > self.tolerance
    
    def register_face(self, encoding):
        """Register a new face"""
        self.known_face_encodings.append(encoding)
        self.known_face_ids.append(self.next_id)
        face_id = self.next_id
        self.next_id += 1
        self.face_count += 1
        return face_id
    
    def identify_face(self, encoding):
        """Identify which known face this is"""
        if len(self.known_face_encodings) == 0:
            return None
        
        distances = face_recognition.face_distance(self.known_face_encodings, encoding)
        min_idx = np.argmin(distances)
        min_distance = distances[min_idx]
        
        if min_distance <= self.tolerance:
            return self.known_face_ids[min_idx]
        return None


def _main():

    start_time = time.time()
    
    
    

    wind_name = 'Face Detection with Distinct Face Tracking'
    
    # Determine if this is camera mode or screen mode
    is_camera_mode = not args.image and not args.video and not args.screen
    is_screen_mode = args.screen
    
    # Enable display by default for camera mode and screen mode
    show_display = args.display or is_camera_mode or is_screen_mode
    
    if show_display:
        cv2.namedWindow(wind_name, cv2.WINDOW_NORMAL)

    output_file = ''
    face_tracker = FaceTracker(tolerance=args.tolerance)
    detected_faces_this_frame = {}

    video_writer = None
    sct = None  # Screen capture object
    monitor = None  # Monitor to capture
    cap = None  # Video capture object
    
    if args.screen:
        # Screen capture mode
        sct = mss.mss()
        monitor = sct.monitors[args.monitor]  # Get specified monitor
        output_file = 'screen_face_detection.avi'
        print(f'[i] Screen capture mode - Monitor {args.monitor}: {monitor["width"]}x{monitor["height"]}')
        
        if args.save_video:
            video_writer = cv2.VideoWriter(os.path.join(args.output_dir, output_file),
                                           cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                           10,  # Lower FPS for screen capture
                                           (monitor['width'], monitor['height']))
    elif args.image:
        if not os.path.isfile(args.image):
            print("[!] ==> Input image file {} doesn't exist".format(args.image))
            sys.exit(1)
        cap = cv2.VideoCapture(args.image)
        output_file = args.image[:-4].rsplit('/')[-1] + '_yoloface.jpg'
    elif args.video:
        if not os.path.isfile(args.video):
            print("[!] ==> Input video file {} doesn't exist".format(args.video))
            sys.exit(1)
        cap = cv2.VideoCapture(args.video)
        output_file = args.video[:-4].rsplit('/')[-1] + '_yoloface.avi'
        # Get the video writer initialized to save the output video (only if requested)
        if args.save_video:
            video_writer = cv2.VideoWriter(os.path.join(args.output_dir, output_file),
                                           cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                           cap.get(cv2.CAP_PROP_FPS), (
                                               round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                               round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    else:
        # Get data from the camera
        cap = cv2.VideoCapture(args.src)
        output_file = 'camera_face_detection.avi'
        # Get the video writer initialized to save the output video (only if requested)
        if args.save_video:
            video_writer = cv2.VideoWriter(os.path.join(args.output_dir, output_file),
                                           cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                           30,  # Default FPS for camera
                                           (
                                               round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                               round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    frame_count = 0
    
    while True:
        # Get frame based on source
        if is_screen_mode:
            # Capture screen
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            # Convert BGRA to BGR (mss captures with alpha channel)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            has_frame = True
        else:
            has_frame, frame = cap.read()

        # Stop the program if reached end of video
        if not has_frame:
            print('[i] ==> Done processing!!!')
            print(f'[i] ==> Total distinct faces detected: {face_tracker.face_count}')
            print('[i] ==> Output file is stored at', os.path.join(args.output_dir, output_file))
            cv2.waitKey(1000)
            break

        frame_count += 1
        
        # Skip frames for faster processing (in fast mode)
        # Still read frames but only process every Nth frame
        if args.fast_mode and args.skip_frames > 1:
            if frame_count % args.skip_frames != 0:
                continue  # Skip this frame entirely
        
        detected_faces_this_frame = {}
        
        # Keep a clean copy of the frame for saving faces (without rectangles)
        clean_frame = frame.copy()

        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                     [0, 0, 0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(get_outputs_names(net))

        # Remove the bounding boxes with low confidence
        faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
        
        # Only print frame updates occasionally
        if frame_count % 50 == 0:
            print(f'[Frame {frame_count}] # detected faces: {len(faces)}')

        # Process each detected face
        new_faces_in_frame = 0
        detected_faces_in_frame = set()
        
        for face_coords in faces:
            # post_process returns [left, top, width, height]
            left, top, width, height = int(face_coords[0]), int(face_coords[1]), int(face_coords[2]), int(face_coords[3])
            x1, y1, x2, y2 = left, top, left + width, top + height
            
            # Get face encoding (skip in fast mode)
            if args.fast_mode or args.image:
                # Fast mode or image: use position-based tracking (no face encoding)
                current_box = (x1, y1, x2, y2)
                
                # Try to match with existing tracked face by position
                face_id = face_tracker.find_matching_face_by_position(current_box)
                
                if face_id is None:
                    # This is a new face - register it
                    face_id = face_tracker.next_id
                    face_tracker.next_id += 1
                    face_tracker.face_count += 1
                    print(f'  [NEW] Face ID #{face_id} detected at frame {frame_count}')
                
                # Update position and last seen
                face_tracker.face_positions[face_id] = current_box
                face_tracker.face_last_seen[face_id] = frame_count
                detected_faces_in_frame.add(face_id)
                
                # For images, save faces immediately; for video, collect samples
                if args.save_faces:
                    margin = 20
                    face_x1 = max(0, x1 - margin)
                    face_y1 = max(0, y1 - margin)
                    face_x2 = min(frame.shape[1], x2 + margin)
                    face_y2 = min(frame.shape[0], y2 + margin)
                    
                    face_img = clean_frame[face_y1:face_y2, face_x1:face_x2].copy()
                    
                    if face_img.size > 0:
                        if args.image:
                            # For single image, save immediately
                            face_path = os.path.join(args.output_dir, 'distinct_faces', 
                                                    f'face_id_{face_id:03d}.jpg')
                            try:
                                success = cv2.imwrite(face_path, face_img)
                                if success:
                                    print(f'  [SAVED] Face ID #{face_id} saved to {face_path}')
                            except Exception as e:
                                print(f'  [ERROR] Failed to save face: {e}')
                        else:
                            # For video/webcam, collect samples
                            # Always collect first sample for new faces
                            is_new = face_id not in face_tracker.face_samples or len(face_tracker.face_samples.get(face_id, [])) == 0
                            
                            if is_new or True:  # Now we collect every processed frame since we skip at frame level
                                quality = face_tracker.add_face_sample(face_id, face_img, frame_count)
                                samples_count = len(face_tracker.face_samples.get(face_id, []))
                                if samples_count <= 5 or samples_count % 10 == 0:
                                    print(f'  [SAMPLE] Face ID #{face_id} - Sample {samples_count} (Quality: {quality:.2f})')
            else:
                # Normal mode with face encoding for duplicate detection
                encoding = face_tracker.get_face_encoding(frame, (x1, y1, x2, y2))
                
                if encoding is None:
                    continue
                
                # Check if it's a new distinct face
                face_id = face_tracker.identify_face(encoding)
                is_new_face = (face_id is None)
                
                if is_new_face:
                    # This is a new distinct face
                    face_id = face_tracker.register_face(encoding)
                
                face_tracker.face_last_seen[face_id] = frame_count
                
                # Collect samples for this face
                margin = 20
                face_x1 = max(0, x1 - margin)
                face_y1 = max(0, y1 - margin)
                face_x2 = min(frame.shape[1], x2 + margin)
                face_y2 = min(frame.shape[0], y2 + margin)
                
                face_img = clean_frame[face_y1:face_y2, face_x1:face_x2].copy()
                
                if face_img.size > 0 and args.save_faces:
                    samples_count = len(face_tracker.face_samples.get(face_id, []))
                    
                    if samples_count < face_tracker.min_samples_to_collect:
                        quality = face_tracker.add_face_sample(face_id, face_img, frame_count)
                        if is_new_face:
                            print(f'  [NEW] Face ID #{face_id} detected!')
                        print(f'  [SAMPLE] Face ID #{face_id} - Sample {samples_count + 1}/{face_tracker.min_samples_to_collect} (Quality: {quality:.2f})')
                    elif face_id not in face_tracker.face_best:
                        # We've collected enough samples, save the best one
                        best_img, best_score, best_frame = face_tracker.get_best_face_image(face_id)
                        if best_img is not None:
                            face_path = os.path.join(args.output_dir, 'distinct_faces', 
                                                    f'face_id_{face_id:03d}_best.jpg')
                            try:
                                success = cv2.imwrite(face_path, best_img)
                                if success:
                                    face_tracker.face_best[face_id] = (face_path, best_score)
                                    print(f'  [BEST] Face ID #{face_id} - Best image saved (Quality: {best_score:.2f})')
                                    face_tracker.face_samples[face_id] = []
                            except Exception as e:
                                print(f'  [ERROR] Failed to save best face: {e}')
                
                detected_faces_in_frame.add(face_id)
            
            detected_faces_this_frame[face_id] = (x1, y1, x2, y2)
            
            # Draw bounding box with face ID
            color = COLOR_RED if face_id < 5 else COLOR_GREEN
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'Face ID: {face_id}', (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Check for faces that haven't been detected for a while
        faces_to_save = []
        for face_id in list(face_tracker.face_samples.keys()):
            if face_id not in face_tracker.face_best and face_id not in detected_faces_in_frame:
                frames_since_seen = frame_count - face_tracker.face_last_seen.get(face_id, frame_count)
                samples_count = len(face_tracker.face_samples[face_id])
                
                # Save if we've waited long enough or have some samples
                if frames_since_seen >= face_tracker.timeout_frames and samples_count > 0:
                    faces_to_save.append(face_id)
        
        # Save best images for timed-out faces
        for face_id in faces_to_save:
            best_img, best_score, best_frame = face_tracker.get_best_face_image(face_id)
            if best_img is not None:
                face_path = os.path.join(args.output_dir, 'distinct_faces', 
                                        f'face_id_{face_id:03d}_best.jpg')
                try:
                    success = cv2.imwrite(face_path, best_img)
                    if success:
                        face_tracker.face_best[face_id] = (face_path, best_score)
                        samples_count = len(face_tracker.face_samples.get(face_id, []))
                        print(f'  [TIMEOUT] Face ID #{face_id} - Saved best from {samples_count} samples (Quality: {best_score:.2f})')
                        # Clear samples to free memory
                        face_tracker.face_samples[face_id] = []
                except Exception as e:
                    print(f'  [ERROR] Failed to save timeout face: {e}')

        # Display statistics on frame
        if show_display:
            info = [
                ('Total Distinct Faces', f'{face_tracker.face_count}'),
                ('Faces in Frame', f'{len(detected_faces_this_frame)}'),
                ('New Faces', f'{new_faces_in_frame}'),
            ]

            for (i, (txt, val)) in enumerate(info):
                text = f'{txt}: {val}'
                cv2.putText(frame, text, (10, (i * 25) + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_RED, 2)

        # Save the output video to file (only if requested and not an image)
        if args.image:
            cv2.imwrite(os.path.join(args.output_dir, output_file), frame.astype(np.uint8))
        elif args.save_video and video_writer:
            video_writer.write(frame.astype(np.uint8))

        if show_display:
            cv2.imshow(wind_name, frame)

            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                print('[i] ==> Interrupted by user!')
                break
        elif is_camera_mode:
            # Camera mode without display still needs a way to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('[i] ==> Interrupted by user!')
                break

    # Save any remaining unsaved faces at end of video
    print('\n[i] Saving remaining unsaved faces...')
    for face_id in list(face_tracker.face_samples.keys()):
        if face_id not in face_tracker.face_best:
            samples_count = len(face_tracker.face_samples.get(face_id, []))
            if samples_count > 0:
                best_result = face_tracker.get_best_face_image(face_id)
                if best_result is not None:
                    best_img, best_score, best_frame = best_result
                    face_path = os.path.join(args.output_dir, 'distinct_faces', 
                                            f'face_id_{face_id:03d}_best.jpg')
                    try:
                        success = cv2.imwrite(face_path, best_img)
                        if success:
                            face_tracker.face_best[face_id] = (face_path, best_score)
                            print(f'  [END] Face ID #{face_id} - Saved best from {samples_count} samples (Quality: {best_score:.2f})')
                            face_tracker.face_samples[face_id] = []
                    except Exception as e:
                        print(f'  [ERROR] Failed to save face: {e}')
    
    # Cleanup
    if cap is not None:
        cap.release()
    if sct is not None:
        sct.close()
    if show_display:
        cv2.destroyAllWindows()
    if video_writer:
        video_writer.release()

    # Save face tracker data
    tracker_path = os.path.join(args.output_dir, 'face_tracker.pkl')
    with open(tracker_path, 'wb') as f:
        pickle.dump(face_tracker, f)
    
    print(f'==> Face tracker saved to {tracker_path}')
    print(f'==> Total distinct faces detected: {face_tracker.face_count}')
    print('==> All done!')
    # At the end, before cleanup:
    elapsed_time = time.time() - start_time
    print(f'[i] Total execution time: {elapsed_time:.2f} seconds')
    print('***********************************************************')


if __name__ == '__main__':
    _main()
