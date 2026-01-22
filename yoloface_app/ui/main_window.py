"""
Main Window - The primary application window
"""

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSlider, QSpinBox, QCheckBox, QScrollArea, QFrame,
    QSplitter, QFileDialog, QTextEdit, QGroupBox, QGridLayout,
    QProgressBar, QSizePolicy, QMessageBox, QStackedWidget,
    QGraphicsDropShadowEffect
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve, QPoint
from PyQt5.QtGui import QIcon, QFont, QColor, QTextCharFormat

from .styles import MAIN_STYLE, LOG_COLORS
from .face_card import FaceCard, FaceCardCompact
from .video_display import VideoDisplay, ImageDisplay
from yoloface_app.core.detector import DetectionWorker

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class SettingsSidebar(QFrame):
    """Overlay settings sidebar"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("settingsSidebar")
        self.setFixedWidth(300)
        self.setStyleSheet("""
            QFrame#settingsSidebar {
                background-color: rgba(30, 30, 46, 0.98);
                border-left: 2px solid #45475a;
            }
        """)
        
        # Add shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30)
        shadow.setColor(QColor(0, 0, 0, 150))
        shadow.setOffset(-5, 0)
        self.setGraphicsEffect(shadow)
        
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)
        
        # Header with close button
        header = QHBoxLayout()
        title = QLabel("⚙️ Settings")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #cdd6f4;")
        self.close_btn = QPushButton("✕")
        self.close_btn.setFixedSize(32, 32)
        self.close_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #6c7086;
                font-size: 18px;
                border: none;
            }
            QPushButton:hover {
                color: #f38ba8;
            }
        """)
        self.close_btn.setCursor(Qt.PointingHandCursor)
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.close_btn)
        layout.addLayout(header)
        
        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("background-color: #45475a;")
        sep.setMaximumHeight(1)
        layout.addWidget(sep)
        
        # Tolerance slider
        tolerance_layout = QVBoxLayout()
        tolerance_header = QHBoxLayout()
        tolerance_label = QLabel("Face Tolerance:")
        self.tolerance_value = QLabel("0.60")
        self.tolerance_value.setStyleSheet("color: #89b4fa; font-weight: bold;")
        tolerance_header.addWidget(tolerance_label)
        tolerance_header.addStretch()
        tolerance_header.addWidget(self.tolerance_value)
        
        self.tolerance_slider = QSlider(Qt.Horizontal)
        self.tolerance_slider.setMinimum(0)
        self.tolerance_slider.setMaximum(100)
        self.tolerance_slider.setValue(50)
        self.tolerance_slider.setTickPosition(QSlider.TicksBelow)
        self.tolerance_slider.setTickInterval(10)
        
        tolerance_layout.addLayout(tolerance_header)
        tolerance_layout.addWidget(self.tolerance_slider)
        layout.addLayout(tolerance_layout)
        
        # Skip frames spinbox
        skip_layout = QHBoxLayout()
        skip_label = QLabel("Skip Frames:")
        self.skip_spinbox = QSpinBox()
        self.skip_spinbox.setMinimum(1)
        self.skip_spinbox.setMaximum(30)
        self.skip_spinbox.setValue(10)
        self.skip_spinbox.setMinimumWidth(80)
        skip_layout.addWidget(skip_label)
        skip_layout.addStretch()
        skip_layout.addWidget(self.skip_spinbox)
        layout.addLayout(skip_layout)
        
        # Sample timeout
        timeout_layout = QHBoxLayout()
        timeout_label = QLabel("Save Timeout (s):")
        self.timeout_spinbox = QSpinBox()
        self.timeout_spinbox.setMinimum(1)
        self.timeout_spinbox.setMaximum(60)
        self.timeout_spinbox.setValue(3)
        self.timeout_spinbox.setMinimumWidth(80)
        timeout_layout.addWidget(timeout_label)
        timeout_layout.addStretch()
        timeout_layout.addWidget(self.timeout_spinbox)
        layout.addLayout(timeout_layout)
        
        # Separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet("background-color: #45475a;")
        sep2.setMaximumHeight(1)
        layout.addWidget(sep2)
        
        # Checkboxes
        self.check_save_output = QCheckBox("Save annotated output")
        self.check_save_output.setChecked(True)
        
        self.check_save_faces = QCheckBox("Save distinct face images")
        self.check_save_faces.setChecked(True)
        
        self.check_show_boxes = QCheckBox("Show detection boxes")
        self.check_show_boxes.setChecked(True)
        
        self.check_fast_mode = QCheckBox("Fast mode (IOU tracking)")
        self.check_fast_mode.setChecked(True)
        
        for check in [self.check_save_output, self.check_save_faces, 
                     self.check_show_boxes, self.check_fast_mode]:
            layout.addWidget(check)
            
        layout.addStretch()


class MainWindow(QMainWindow):
    """Main application window for YOLOFace Tracker"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOFace Tracker")
        self.setMinimumSize(1280, 800)
        self.setStyleSheet(MAIN_STYLE)
        
        # State
        self.detection_worker = None
        self.face_cards = {}  # face_id -> FaceCard
        self.current_mode = None  # 'image', 'video', 'webcam'
        self.is_processing = False
        self.is_paused = False
        self.source_path = None
        
        # Webcam preview state
        self.webcam_preview_active = False
        self.preview_cap = None
        self.preview_timer = None
        
        self.setup_ui()
        self.connect_signals()
        
        # Start fullscreen
        QTimer.singleShot(100, self.showMaximized)
        
    def setup_ui(self):
        """Setup the main UI layout"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # === LEFT SECTION - Main content (2/3 width) ===
        left_section = QWidget()
        left_layout = QVBoxLayout(left_section)
        left_layout.setContentsMargins(16, 16, 8, 16)
        left_layout.setSpacing(12)
        
        # -- Top bar with source buttons and settings toggle --
        top_bar = QFrame()
        top_bar.setObjectName("sourceFrame")
        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setContentsMargins(16, 12, 16, 12)
        
        source_label = QLabel("Source:")
        source_label.setStyleSheet("color: #cdd6f4; font-weight: bold; font-size: 14px;")
        top_bar_layout.addWidget(source_label)
        
        self.btn_image = QPushButton("📷  Image")
        self.btn_video = QPushButton("🎬  Video")
        self.btn_webcam = QPushButton("📹  Webcam")
        
        for btn in [self.btn_image, self.btn_video, self.btn_webcam]:
            btn.setMinimumHeight(40)
            btn.setMinimumWidth(120)
            btn.setCursor(Qt.PointingHandCursor)
            top_bar_layout.addWidget(btn)
            
        top_bar_layout.addStretch()
        
        # Settings toggle button
        self.btn_settings = QPushButton("⚙️  Settings")
        self.btn_settings.setMinimumHeight(40)
        self.btn_settings.setCursor(Qt.PointingHandCursor)
        top_bar_layout.addWidget(self.btn_settings)
        
        left_layout.addWidget(top_bar)
        
        # -- Video/Image display --
        self.display_stack = QStackedWidget()
        
        self.video_display = VideoDisplay()
        self.image_display = ImageDisplay()
        self.loading_widget = self.create_loading_widget()
        
        self.display_stack.addWidget(self.video_display)   # Index 0
        self.display_stack.addWidget(self.image_display)   # Index 1
        self.display_stack.addWidget(self.loading_widget)  # Index 2
        
        left_layout.addWidget(self.display_stack, stretch=2)
        
        # -- Control buttons --
        controls_frame = QFrame()
        controls_frame.setObjectName("controlsFrame")
        controls_layout = QHBoxLayout(controls_frame)
        controls_layout.setContentsMargins(12, 10, 12, 10)
        controls_layout.setSpacing(10)
        
        self.btn_start = QPushButton("▶ Start")
        self.btn_start.setObjectName("startButton")
        self.btn_start.setFixedHeight(36)
        self.btn_start.setFixedWidth(100)
        self.btn_start.setEnabled(False)
        
        self.btn_pause = QPushButton("⏸ Pause")
        self.btn_pause.setFixedHeight(36)
        self.btn_pause.setFixedWidth(100)
        self.btn_pause.setEnabled(False)
        
        self.btn_stop = QPushButton("⏹ Stop")
        self.btn_stop.setObjectName("stopButton")
        self.btn_stop.setFixedHeight(36)
        self.btn_stop.setFixedWidth(100)
        self.btn_stop.setEnabled(False)
        
        for btn in [self.btn_start, self.btn_pause, self.btn_stop]:
            btn.setCursor(Qt.PointingHandCursor)
            controls_layout.addWidget(btn)
            
        # Spacer
        controls_layout.addSpacing(20)
        
        # Stats label
        self.stats_label = QLabel("")
        self.stats_label.setStyleSheet("color: #a6e3a1; font-size: 12px;")
        controls_layout.addWidget(self.stats_label)
        
        controls_layout.addStretch()
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.setFixedHeight(18)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        controls_layout.addWidget(self.progress_bar)
        
        left_layout.addWidget(controls_frame)
        
        # -- Log output --
        log_group = QGroupBox("Log Output")
        log_group.setMaximumHeight(180)
        log_layout = QVBoxLayout(log_group)
        
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setFont(QFont("Consolas", 10))
        self.log_output.setStyleSheet("""
            QTextEdit {
                background-color: #11111b;
                color: #cdd6f4;
                border: 1px solid #313244;
                border-radius: 6px;
                padding: 8px;
            }
        """)
        log_layout.addWidget(self.log_output)
        
        left_layout.addWidget(log_group)
        
        # === RIGHT SECTION - Detected Faces (1/3 width) ===
        right_section = QWidget()
        right_section.setMinimumWidth(320)
        right_section.setMaximumWidth(400)
        right_layout = QVBoxLayout(right_section)
        right_layout.setContentsMargins(8, 16, 16, 16)
        right_layout.setSpacing(12)
        
        # -- Detected Faces section --
        faces_group = QGroupBox("Detected Faces")
        faces_layout = QVBoxLayout(faces_group)
        
        self.faces_scroll = QScrollArea()
        self.faces_scroll.setWidgetResizable(True)
        self.faces_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.faces_scroll.setStyleSheet("""
            QScrollArea {
                background-color: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background-color: #1e1e2e;
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background-color: #45475a;
                border-radius: 5px;
                min-height: 20px;
            }
        """)
        
        self.faces_container = QWidget()
        self.faces_list = QVBoxLayout(self.faces_container)
        self.faces_list.setSpacing(8)
        self.faces_list.setContentsMargins(4, 4, 4, 4)
        self.faces_list.addStretch()
        self.faces_scroll.setWidget(self.faces_container)
        
        # Empty state label
        self.faces_empty_label = QLabel("No faces detected yet")
        self.faces_empty_label.setAlignment(Qt.AlignCenter)
        self.faces_empty_label.setStyleSheet("color: #6c7086; font-size: 14px; padding: 40px;")
        self.faces_list.insertWidget(0, self.faces_empty_label)
        
        faces_layout.addWidget(self.faces_scroll)
        
        # Clear faces button
        self.btn_clear_faces = QPushButton("🗑  Clear All Faces")
        self.btn_clear_faces.setEnabled(False)
        self.btn_clear_faces.setCursor(Qt.PointingHandCursor)
        faces_layout.addWidget(self.btn_clear_faces)
        
        right_layout.addWidget(faces_group)
        
        # Add sections to main layout
        main_layout.addWidget(left_section, stretch=2)
        main_layout.addWidget(right_section, stretch=1)
        
        # === Settings Sidebar (Overlay) ===
        self.settings_sidebar = SettingsSidebar(self)
        self.settings_sidebar.hide()
        self.settings_sidebar.close_btn.clicked.connect(self.toggle_settings)
        self.settings_sidebar.tolerance_slider.valueChanged.connect(self.update_tolerance)
        
    def create_loading_widget(self):
        """Create a loading indicator widget"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignCenter)
        
        spinner_label = QLabel("⏳")
        spinner_label.setStyleSheet("font-size: 48px;")
        spinner_label.setAlignment(Qt.AlignCenter)
        
        loading_text = QLabel("Loading...")
        loading_text.setStyleSheet("color: #cdd6f4; font-size: 16px;")
        loading_text.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(spinner_label)
        layout.addWidget(loading_text)
        
        return widget
        
    def connect_signals(self):
        """Connect UI signals to slots"""
        # Source buttons
        self.btn_image.clicked.connect(lambda: self.select_source('image'))
        self.btn_video.clicked.connect(lambda: self.select_source('video'))
        self.btn_webcam.clicked.connect(lambda: self.select_source('webcam'))
        
        # Control buttons
        self.btn_start.clicked.connect(self.start_detection)
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_stop.clicked.connect(self.stop_detection)
        self.btn_clear_faces.clicked.connect(self.clear_faces)
        
        # Settings
        self.btn_settings.clicked.connect(self.toggle_settings)
        
    def toggle_settings(self):
        """Toggle settings sidebar visibility"""
        if self.settings_sidebar.isVisible():
            self.settings_sidebar.hide()
        else:
            # Position sidebar on right edge
            self.settings_sidebar.move(
                self.width() - self.settings_sidebar.width(),
                0
            )
            self.settings_sidebar.setFixedHeight(self.height())
            self.settings_sidebar.show()
            self.settings_sidebar.raise_()
            
    def resizeEvent(self, event):
        """Handle window resize"""
        super().resizeEvent(event)
        # Update settings sidebar position if visible
        if self.settings_sidebar.isVisible():
            self.settings_sidebar.move(
                self.width() - self.settings_sidebar.width(),
                0
            )
            self.settings_sidebar.setFixedHeight(self.height())
        
    def update_tolerance(self, value):
        """Update tolerance display"""
        tolerance = value / 100.0
        self.settings_sidebar.tolerance_value.setText(f"{tolerance:.2f}")
        
    def log_message(self, message, level="INFO"):
        """Add a message to the log output"""
        color = LOG_COLORS.get(level, LOG_COLORS.get("info", "#cdd6f4"))
        timestamp = self.get_timestamp()
        
        # Format with HTML for colors
        formatted = f'<span style="color: #6c7086;">[{timestamp}]</span> <span style="color: {color};">{message}</span>'
        self.log_output.append(formatted)
        
        # Scroll to bottom
        scrollbar = self.log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def get_timestamp(self):
        """Get current timestamp string"""
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")
        
    def select_source(self, source_type):
        """Handle source selection"""
        # Stop any webcam preview if switching sources
        self.stop_webcam_preview()
        
        self.current_mode = source_type
        
        # Clear video display
        self.video_display.set_placeholder("Select a source...")
        self.image_display.set_image(None)
        
        # Update button styles
        for btn, mode in [(self.btn_image, 'image'), 
                          (self.btn_video, 'video'), 
                          (self.btn_webcam, 'webcam')]:
            if mode == source_type:
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #89b4fa;
                        color: #1e1e2e;
                        font-weight: bold;
                    }
                """)
            else:
                btn.setStyleSheet("")
                
        if source_type == 'image':
            self.select_image()
        elif source_type == 'video':
            self.select_video()
        elif source_type == 'webcam':
            self.setup_webcam()
            
    def select_image(self):
        """Open file dialog for image selection"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Image Files (*.jpg *.jpeg *.png *.bmp *.webp);;All Files (*)"
        )
        if file_path:
            self.source_path = file_path
            self.display_stack.setCurrentIndex(1)  # Image display
            self.image_display.set_image(file_path)
            self.btn_start.setEnabled(True)
            self.log_message(f"Selected image: {os.path.basename(file_path)}", "info")
        else:
            self.current_mode = None
            
    def select_video(self):
        """Open file dialog for video selection"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*)"
        )
        if file_path:
            self.source_path = file_path
            self.display_stack.setCurrentIndex(0)  # Video display
            
            # Show first frame of video as preview
            import cv2
            cap = cv2.VideoCapture(file_path)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    self.video_display.set_frame(frame)
                else:
                    self.video_display.set_placeholder(f"Ready: {os.path.basename(file_path)}")
                cap.release()
            else:
                self.video_display.set_placeholder(f"Ready: {os.path.basename(file_path)}")
            
            self.btn_start.setEnabled(True)
            self.log_message(f"Selected video: {os.path.basename(file_path)}", "info")
        else:
            self.current_mode = None
            
    def setup_webcam(self):
        """Setup webcam source and show preview without model"""
        self.source_path = 0  # Default webcam
        self.display_stack.setCurrentIndex(0)  # Video display
        self.btn_start.setEnabled(True)
        self.log_message("Webcam selected - opening preview...", "info")
        
        # Start webcam preview (without detection)
        self.start_webcam_preview()
        
    def start_webcam_preview(self):
        """Start webcam preview without running the detection model"""
        import cv2
        self.webcam_preview_active = True
        self.preview_cap = cv2.VideoCapture(0)
        
        if not self.preview_cap.isOpened():
            self.video_display.set_placeholder("Failed to open webcam")
            self.log_message("Failed to open webcam", "error")
            return
            
        # Create timer for preview frames
        self.preview_timer = QTimer()
        self.preview_timer.timeout.connect(self.update_webcam_preview)
        self.preview_timer.start(33)  # ~30 fps
        
    def update_webcam_preview(self):
        """Update webcam preview frame"""
        if hasattr(self, 'preview_cap') and self.preview_cap is not None:
            ret, frame = self.preview_cap.read()
            if ret:
                self.video_display.set_frame(frame)
                
    def stop_webcam_preview(self):
        """Stop the webcam preview"""
        if hasattr(self, 'preview_timer') and self.preview_timer is not None:
            self.preview_timer.stop()
            self.preview_timer = None
        if hasattr(self, 'preview_cap') and self.preview_cap is not None:
            self.preview_cap.release()
            self.preview_cap = None
        self.webcam_preview_active = False
        
    def start_detection(self):
        """Start the detection process"""
        if self.is_processing:
            return
            
        # Stop webcam preview if active
        self.stop_webcam_preview()
            
        self.is_processing = True
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setValue(0)
        
        # Get settings from sidebar
        settings = {
            'tolerance': self.settings_sidebar.tolerance_slider.value() / 100.0,
            'skip_frames': self.settings_sidebar.skip_spinbox.value(),
            'sample_timeout': self.settings_sidebar.timeout_spinbox.value(),
            'save_output': self.settings_sidebar.check_save_output.isChecked(),
            'save_faces': self.settings_sidebar.check_save_faces.isChecked(),
            'show_boxes': self.settings_sidebar.check_show_boxes.isChecked(),
            'fast_mode': self.settings_sidebar.check_fast_mode.isChecked()
        }
        
        # Create and start worker
        self.detection_worker = DetectionWorker(
            self.current_mode,
            self.source_path,
            settings
        )
        
        # Connect worker signals
        self.detection_worker.frame_ready.connect(self.on_frame_ready)
        self.detection_worker.face_detected.connect(self.on_face_detected)
        self.detection_worker.new_face.connect(self.on_new_face)
        self.detection_worker.sample_added.connect(self.on_sample_added)
        self.detection_worker.face_saved.connect(self.on_face_saved)
        self.detection_worker.log_message.connect(self.log_message)
        self.detection_worker.progress_update.connect(self.on_progress_update)
        self.detection_worker.stats_update.connect(self.on_stats_update)
        self.detection_worker.processing_complete.connect(self.on_processing_complete)
        self.detection_worker.error.connect(self.on_error)
        
        self.detection_worker.start()
        self.log_message(f"Started {self.current_mode} detection", "success")
        
    def toggle_pause(self):
        """Toggle pause state"""
        if self.detection_worker:
            self.is_paused = not self.is_paused
            self.detection_worker.set_paused(self.is_paused)
            
            if self.is_paused:
                self.btn_pause.setText("▶  Resume")
                self.log_message("Detection paused", "warning")
            else:
                self.btn_pause.setText("⏸  Pause")
                self.log_message("Detection resumed", "info")
                
    def stop_detection(self):
        """Stop the detection process"""
        if self.detection_worker:
            self.detection_worker.stop()
            self.detection_worker.wait()
            self.detection_worker = None
            
        self.is_processing = False
        self.is_paused = False
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_pause.setText("⏸  Pause")
        self.log_message("Detection stopped", "warning")
        
    def on_frame_ready(self, frame):
        """Handle new frame from worker"""
        if self.current_mode == 'image':
            # For images, show the annotated frame in image display
            self.display_stack.setCurrentIndex(1)  # Switch to image display
            self.image_display.set_image(frame)
        else:
            self.video_display.set_frame(frame)
        
    def on_face_detected(self, face_id, face_image, box):
        """Handle face detection event"""
        # Update existing face card if exists
        if face_id in self.face_cards:
            self.face_cards[face_id].update_image(face_image)
            
    def on_new_face(self, face_id, face_image):
        """Handle new face detected"""
        # Remove empty label if present
        if self.faces_empty_label.isVisible():
            self.faces_empty_label.hide()
            
        # Create new face card (compact, one per row)
        card = FaceCardCompact(face_id)
        card.update_image(face_image)
        card.delete_requested.connect(self.delete_face)
        
        # Add to list (insert before stretch)
        self.faces_list.insertWidget(self.faces_list.count() - 1, card)
        
        self.face_cards[face_id] = card
        card.animate_entrance()
        
        self.btn_clear_faces.setEnabled(True)
        self.log_message(f"New face detected: ID {face_id}", "success")
        
    def on_sample_added(self, face_id, sample_count):
        """Handle sample added to face"""
        if face_id in self.face_cards:
            self.face_cards[face_id].update_samples(sample_count)
            
    def on_face_saved(self, face_id, file_path):
        """Handle face saved to file"""
        if face_id in self.face_cards:
            self.face_cards[face_id].mark_saved(file_path)
        self.log_message(f"Face {face_id} saved: {os.path.basename(file_path)}", "success")
        
    def on_progress_update(self, progress):
        """Update progress bar"""
        self.progress_bar.setValue(int(progress))
        
    def on_stats_update(self, total_faces, current_frame, fps):
        """Update stats display"""
        self.stats_label.setText(f"Faces: {total_faces} | Frame: {current_frame} | FPS: {fps:.1f}")
        
    def on_processing_complete(self, summary):
        """Handle processing complete"""
        self.is_processing = False
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setValue(100)
        
        self.log_message(f"Processing complete! {summary}", "success")
        
    def on_error(self, error_msg):
        """Handle error from worker"""
        self.log_message(f"Error: {error_msg}", "error")
        QMessageBox.critical(self, "Error", error_msg)
        self.stop_detection()
        
    def delete_face(self, face_id):
        """Delete a face card and its saved file from disk"""
        if face_id in self.face_cards:
            card = self.face_cards.pop(face_id)
            
            # Delete the saved file from disk if it exists
            file_path = card.get_file_path()
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    self.log_message(f"Deleted file: {os.path.basename(file_path)}", "warning")
                except Exception as e:
                    self.log_message(f"Failed to delete file: {e}", "error")
            
            self.faces_list.removeWidget(card)
            card.deleteLater()
            
            # Show empty label if no faces left
            if not self.face_cards:
                self.faces_empty_label.show()
                self.btn_clear_faces.setEnabled(False)
                
            self.log_message(f"Removed face {face_id}", "warning")
            
    def clear_faces(self):
        """Clear all face cards"""
        for face_id in list(self.face_cards.keys()):
            self.delete_face(face_id)
            
    def closeEvent(self, event):
        """Handle window close"""
        # Stop webcam preview
        self.stop_webcam_preview()
        
        if self.detection_worker:
            self.detection_worker.stop()
            self.detection_worker.wait()
        event.accept()
