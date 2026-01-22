"""
Face Card Widget - Displays individual detected faces with animation
"""

from PyQt5.QtWidgets import (QFrame, QLabel, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QSizePolicy, QGraphicsOpacityEffect)
from PyQt5.QtCore import Qt, QPropertyAnimation, QSize, QTimer, pyqtProperty, QEasingCurve, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
import numpy as np


class FaceCard(QFrame):
    """Widget displaying a detected face with info"""
    
    def __init__(self, face_id, face_image, frame_num, quality, parent=None):
        super().__init__(parent)
        self.face_id = face_id
        self.frame_num = frame_num
        self.quality = quality
        self._scale = 1.0
        
        self.setObjectName("faceCard")
        self.setup_ui(face_image)
        
    def setup_ui(self, face_image):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        
        # Face image
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(100, 100)
        self.image_label.setMaximumSize(150, 150)
        self.image_label.setScaledContents(True)
        
        # Convert numpy array to QPixmap
        self.set_image(face_image)
        
        # Info labels
        self.id_label = QLabel(f"ID: {self.face_id}")
        self.id_label.setAlignment(Qt.AlignCenter)
        self.id_label.setStyleSheet("font-weight: bold; color: #89b4fa; font-size: 14px;")
        
        self.frame_label = QLabel(f"Frame: {self.frame_num}")
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.frame_label.setStyleSheet("color: #a6adc8; font-size: 11px;")
        
        self.quality_label = QLabel(f"Quality: {self.quality:.2f}")
        self.quality_label.setAlignment(Qt.AlignCenter)
        
        # Color code quality
        if self.quality >= 0.8:
            self.quality_label.setStyleSheet("color: #a6e3a1; font-size: 11px;")
        elif self.quality >= 0.5:
            self.quality_label.setStyleSheet("color: #f9e2af; font-size: 11px;")
        else:
            self.quality_label.setStyleSheet("color: #f38ba8; font-size: 11px;")
        
        # Delete button
        self.delete_btn = QPushButton("×")
        self.delete_btn.setFixedSize(24, 24)
        self.delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #f38ba8;
                color: #1e1e2e;
                border-radius: 12px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #eba0ac;
            }
        """)
        self.delete_btn.setCursor(Qt.PointingHandCursor)
        
        # Header with delete button
        header = QHBoxLayout()
        header.addStretch()
        header.addWidget(self.delete_btn)
        
        layout.addLayout(header)
        layout.addWidget(self.image_label)
        layout.addWidget(self.id_label)
        layout.addWidget(self.frame_label)
        layout.addWidget(self.quality_label)
        
        self.setFixedSize(170, 220)
        
    def set_image(self, face_image):
        """Set the face image from numpy array"""
        if face_image is None:
            return
            
        if isinstance(face_image, np.ndarray):
            height, width, channel = face_image.shape
            bytes_per_line = 3 * width
            # Convert BGR to RGB
            rgb_image = face_image[:, :, ::-1].copy()
            q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
        else:
            pixmap = face_image
            
        self.image_label.setPixmap(pixmap.scaled(140, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation))
    
    def animate_entrance(self):
        """Animate the card appearing (scale up effect)"""
        # Start small
        self.setFixedSize(50, 65)
        
        # Animate to full size
        self.animation = QTimer(self)
        self.animation.timeout.connect(self._grow_step)
        self._target_width = 170
        self._target_height = 220
        self._current_width = 50
        self._current_height = 65
        self.animation.start(16)  # ~60fps
        
    def _grow_step(self):
        """Animation step for growing"""
        step = 15
        self._current_width = min(self._current_width + step, self._target_width)
        self._current_height = min(self._current_height + int(step * 1.3), self._target_height)
        
        self.setFixedSize(self._current_width, self._current_height)
        
        if self._current_width >= self._target_width and self._current_height >= self._target_height:
            self.animation.stop()
            self.setFixedSize(170, 220)


class FaceCardCompact(QFrame):
    """Compact face card for list view"""
    
    delete_requested = pyqtSignal(int)  # face_id
    
    def __init__(self, face_id, parent=None):
        super().__init__(parent)
        self.face_id = face_id
        self.sample_count = 0
        self.is_saved = False
        self.file_path = None  # Path to saved face image on disk
        
        self.setObjectName("faceCard")
        self.setup_ui()
        
    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)
        
        # Face image
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(60, 60)
        self.image_label.setScaledContents(True)
        self.image_label.setStyleSheet("border-radius: 8px; background-color: #313244;")
        
        # Info section
        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)
        
        self.id_label = QLabel(f"Face ID #{self.face_id}")
        self.id_label.setStyleSheet("font-weight: bold; color: #89b4fa; font-size: 12px;")
        
        self.details_label = QLabel(f"Samples: 0")
        self.details_label.setStyleSheet("color: #a6adc8; font-size: 11px;")
        
        self.status_label = QLabel("Collecting...")
        self.status_label.setStyleSheet("color: #f9e2af; font-size: 10px;")
        
        info_layout.addWidget(self.id_label)
        info_layout.addWidget(self.details_label)
        info_layout.addWidget(self.status_label)
        info_layout.addStretch()
        
        # Delete button
        self.delete_btn = QPushButton("×")
        self.delete_btn.setFixedSize(32, 32)
        self.delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #45475a;
                color: #f38ba8;
                border-radius: 16px;
                font-size: 20px;
                font-weight: bold;
                padding: 0px;
                margin: 0px;
                line-height: 32px;
            }
            QPushButton:hover {
                background-color: #f38ba8;
                color: #1e1e2e;
            }
        """)
        self.delete_btn.setCursor(Qt.PointingHandCursor)
        self.delete_btn.clicked.connect(lambda: self.delete_requested.emit(self.face_id))
        
        layout.addWidget(self.image_label)
        layout.addLayout(info_layout, 1)
        layout.addWidget(self.delete_btn, 0, Qt.AlignVCenter)
        
        self.setFixedHeight(90)
        self.setMinimumWidth(200)
        
    def update_image(self, face_image):
        """Update the face image"""
        if face_image is None:
            return
            
        if isinstance(face_image, np.ndarray):
            height, width = face_image.shape[:2]
            channel = face_image.shape[2] if len(face_image.shape) > 2 else 1
            if channel == 3:
                bytes_per_line = 3 * width
                rgb_image = face_image[:, :, ::-1].copy()
                q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            else:
                q_image = QImage(face_image.data, width, height, width, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_image)
        else:
            pixmap = face_image
            
        self.image_label.setPixmap(pixmap.scaled(60, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
    def update_samples(self, count):
        """Update sample count display"""
        self.sample_count = count
        self.details_label.setText(f"Samples: {count}")
        
    def mark_saved(self, file_path=None):
        """Mark this face as saved and store the file path"""
        self.is_saved = True
        self.file_path = file_path
        self.status_label.setText("✓ Saved")
        self.status_label.setStyleSheet("color: #a6e3a1; font-size: 10px;")
    
    def get_file_path(self):
        """Get the saved file path"""
        return self.file_path
        
    def animate_entrance(self):
        """Animate the card appearing"""
        effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(effect)
        
        self.anim = QPropertyAnimation(effect, b"opacity")
        self.anim.setDuration(300)
        self.anim.setStartValue(0)
        self.anim.setEndValue(1)
        self.anim.setEasingCurve(QEasingCurve.OutCubic)
        self.anim.start()
