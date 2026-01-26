"""
Video Display Widget - Shows video/webcam feed with face detection boxes
"""

from PyQt5.QtWidgets import (QWidget, QLabel, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QSizePolicy, QFrame)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QFont
import numpy as np
import cv2

# Screen size utilities for dynamic scaling
BASE_SCREEN_WIDTH = 1920
BASE_SCREEN_HEIGHT = 1200

def get_screen_size():
    """Return the width and height of the primary screen."""
    try:
        from PyQt5.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            # Create a temporary QApplication if none exists
            app = QApplication([])
        
        screen = app.primaryScreen()
        if screen is None:
            # Fallback to default screen size
            return BASE_SCREEN_WIDTH, BASE_SCREEN_HEIGHT
        
        geometry = screen.geometry()
        return geometry.width(), geometry.height()
    except:
        # Fallback if QApplication is not available
        return BASE_SCREEN_WIDTH, BASE_SCREEN_HEIGHT

def scale_width_percent(percent):
    """Scale a width value based on percentage of screen width."""
    current_width, _ = get_screen_size()
    return int((percent / 100.0) * current_width)

def scale_height_percent(percent):
    """Scale a height value based on percentage of screen height."""
    _, current_height = get_screen_size()
    return int((percent / 100.0) * current_height)

def scale_dimension_percent(percent):
    """Scale a dimension value based on percentage of average screen dimensions."""
    current_width, current_height = get_screen_size()
    avg_dimension = (current_width + current_height) / 2
    base_avg = (BASE_SCREEN_WIDTH + BASE_SCREEN_HEIGHT) / 2
    return int((percent / 100.0) * avg_dimension)


class VideoDisplay(QFrame):
    """Widget for displaying video feed with detection overlays"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("videoFrame")
        self.setMinimumSize(scale_width_percent(20.83), scale_height_percent(16.67))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.setup_ui()
        self._current_frame = None
        self._is_fullscreen = False
        self._original_parent = None
        self._original_geometry = None
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Video display label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(scale_width_percent(10.42), scale_height_percent(12.5))
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #11111b;
                border-radius: 8px;
            }
        """)
        
        # Placeholder text
        self.video_label.setText("No video source")
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #11111b;
                color: #6c7086;
                font-size: 18px;
                border-radius: 8px;
            }
        """)
        
        # Controls overlay
        self.controls_widget = QWidget()
        self.controls_widget.setStyleSheet("background-color: transparent;")
        controls_layout = QHBoxLayout(self.controls_widget)
        controls_layout.setContentsMargins(10, 10, 10, 10)
        
        self.fullscreen_btn = QPushButton("⛶")
        self.fullscreen_btn.setFixedSize(scale_dimension_percent(2.31), scale_dimension_percent(2.31))
        self.fullscreen_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(30, 30, 46, 0.8);
                color: #cdd6f4;
                border-radius: 18px;
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: rgba(69, 71, 90, 0.9);
            }
        """)
        self.fullscreen_btn.clicked.connect(self.toggle_fullscreen)
        self.fullscreen_btn.setCursor(Qt.PointingHandCursor)
        
        controls_layout.addStretch()
        controls_layout.addWidget(self.fullscreen_btn)
        
        layout.addWidget(self.video_label)
        
        # Stats overlay
        self.stats_label = QLabel()
        self.stats_label.setStyleSheet("""
            QLabel {
                background-color: rgba(17, 17, 27, 0.85);
                color: #a6e3a1;
                padding: 8px 12px;
                border-radius: 8px;
                font-size: 12px;
                font-family: 'Consolas', monospace;
            }
        """)
        self.stats_label.hide()
        
    def set_frame(self, frame):
        """Update the display with a new frame (numpy array)"""
        if frame is None:
            return
            
        self._current_frame = frame.copy()
        
        # Convert BGR to RGB and make contiguous copy
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = np.ascontiguousarray(rgb_frame)
        height, width, channel = rgb_frame.shape
        bytes_per_line = 3 * width
        
        # Create QImage with copy of data to prevent memory issues
        q_image = QImage(rgb_frame.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        # Scale to fit label while maintaining aspect ratio
        label_size = self.video_label.size()
        if label_size.width() > 0 and label_size.height() > 0:
            scaled_pixmap = pixmap.scaled(
                label_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.video_label.setPixmap(scaled_pixmap)
        else:
            self.video_label.setPixmap(pixmap)
        
    def set_placeholder(self, text="No video source"):
        """Show placeholder text"""
        self.video_label.clear()
        self.video_label.setText(text)
        self._current_frame = None
        
    def update_stats(self, total_faces, current_frame, faces_in_frame):
        """Update the stats overlay"""
        stats_text = f"Faces: {total_faces} | Frame: {current_frame} | In Frame: {faces_in_frame}"
        self.stats_label.setText(stats_text)
        if not self.stats_label.isVisible():
            self.stats_label.show()
            
    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        if self._is_fullscreen:
            self.exit_fullscreen()
        else:
            self.enter_fullscreen()
            
    def enter_fullscreen(self):
        """Enter fullscreen mode"""
        self._is_fullscreen = True
        self._original_parent = self.parent()
        self._original_geometry = self.geometry()
        
        self.setParent(None)
        self.showFullScreen()
        self.fullscreen_btn.setText("✕")
        
    def exit_fullscreen(self):
        """Exit fullscreen mode"""
        self._is_fullscreen = False
        self.showNormal()
        if self._original_parent:
            self.setParent(self._original_parent)
            self.show()
        self.fullscreen_btn.setText("⛶")
        
    def keyPressEvent(self, event):
        """Handle key press for fullscreen exit"""
        if event.key() == Qt.Key_Escape and self._is_fullscreen:
            self.exit_fullscreen()
        else:
            super().keyPressEvent(event)


class ImageDisplay(QFrame):
    """Widget for displaying a single image with detection boxes"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("videoFrame")
        self.setMinimumSize(scale_width_percent(20.83), scale_height_percent(16.67))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #11111b;
                border-radius: 8px;
            }
        """)
        self.image_label.setText("No image loaded")
        
        layout.addWidget(self.image_label)
        
    def set_image(self, image):
        """Set image from numpy array or file path"""
        if image is None:
            self.image_label.setText("No image loaded")
            return
            
        if isinstance(image, str):
            # Load from file
            pixmap = QPixmap(image)
        elif isinstance(image, np.ndarray):
            # Convert from numpy array
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, channel = rgb_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
        else:
            return
            
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
