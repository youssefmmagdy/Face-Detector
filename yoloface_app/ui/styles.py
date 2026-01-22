"""
Stylesheet definitions for the YOLOFace Tracker Application
"""

MAIN_STYLE = """
QMainWindow {
    background-color: #1e1e2e;
}

QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
}

/* Buttons */
QPushButton {
    background-color: #45475a;
    color: #cdd6f4;
    border: none;
    border-radius: 8px;
    padding: 12px 24px;
    font-size: 14px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #585b70;
}

QPushButton:pressed {
    background-color: #313244;
}

QPushButton:disabled {
    background-color: #313244;
    color: #6c7086;
}

QPushButton#primaryButton {
    background-color: #89b4fa;
    color: #1e1e2e;
}

QPushButton#primaryButton:hover {
    background-color: #b4befe;
}

QPushButton#dangerButton {
    background-color: #f38ba8;
    color: #1e1e2e;
}

QPushButton#dangerButton:hover {
    background-color: #eba0ac;
}

QPushButton#successButton {
    background-color: #a6e3a1;
    color: #1e1e2e;
}

QPushButton#successButton:hover {
    background-color: #94e2d5;
}

/* Labels */
QLabel {
    color: #cdd6f4;
    font-size: 13px;
}

QLabel#titleLabel {
    font-size: 24px;
    font-weight: bold;
    color: #89b4fa;
}

QLabel#subtitleLabel {
    font-size: 16px;
    color: #a6adc8;
}

QLabel#sectionLabel {
    font-size: 14px;
    font-weight: bold;
    color: #f5c2e7;
    padding: 8px 0;
}

QLabel#statsLabel {
    font-size: 28px;
    font-weight: bold;
    color: #a6e3a1;
}

/* Sliders */
QSlider::groove:horizontal {
    border: none;
    height: 8px;
    background: #313244;
    border-radius: 4px;
}

QSlider::handle:horizontal {
    background: #89b4fa;
    border: none;
    width: 18px;
    height: 18px;
    margin: -5px 0;
    border-radius: 9px;
}

QSlider::handle:horizontal:hover {
    background: #b4befe;
}

QSlider::sub-page:horizontal {
    background: #89b4fa;
    border-radius: 4px;
}

/* SpinBox */
QSpinBox, QDoubleSpinBox {
    background-color: #313244;
    color: #cdd6f4;
    border: 2px solid #45475a;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 14px;
}

QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #89b4fa;
}

QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    background-color: #45475a;
    border: none;
    width: 20px;
}

/* CheckBox */
QCheckBox {
    color: #cdd6f4;
    font-size: 13px;
    spacing: 8px;
}

QCheckBox::indicator {
    width: 20px;
    height: 20px;
    border-radius: 4px;
    border: 2px solid #45475a;
    background-color: #313244;
}

QCheckBox::indicator:checked {
    background-color: #89b4fa;
    border-color: #89b4fa;
}

QCheckBox::indicator:hover {
    border-color: #89b4fa;
}

/* ScrollArea */
QScrollArea {
    border: none;
    background-color: transparent;
}

QScrollBar:vertical {
    background-color: #313244;
    width: 12px;
    border-radius: 6px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background-color: #45475a;
    border-radius: 6px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background-color: #585b70;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

QScrollBar:horizontal {
    background-color: #313244;
    height: 12px;
    border-radius: 6px;
}

QScrollBar::handle:horizontal {
    background-color: #45475a;
    border-radius: 6px;
    min-width: 30px;
}

/* TextEdit / Log Area */
QTextEdit {
    background-color: #11111b;
    color: #a6e3a1;
    border: 2px solid #313244;
    border-radius: 8px;
    padding: 10px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
}

/* Progress Bar */
QProgressBar {
    background-color: #313244;
    border: none;
    border-radius: 8px;
    height: 20px;
    text-align: center;
    color: #1e1e2e;
    font-weight: bold;
}

QProgressBar::chunk {
    background-color: #89b4fa;
    border-radius: 8px;
}

/* GroupBox */
QGroupBox {
    background-color: #181825;
    border: 2px solid #313244;
    border-radius: 12px;
    margin-top: 16px;
    padding: 16px;
    font-size: 14px;
    font-weight: bold;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 16px;
    padding: 0 8px;
    color: #f5c2e7;
}

/* Tab Widget */
QTabWidget::pane {
    border: 2px solid #313244;
    border-radius: 8px;
    background-color: #181825;
}

QTabBar::tab {
    background-color: #313244;
    color: #a6adc8;
    padding: 12px 24px;
    margin-right: 4px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
}

QTabBar::tab:selected {
    background-color: #89b4fa;
    color: #1e1e2e;
}

QTabBar::tab:hover:!selected {
    background-color: #45475a;
}

/* Frame */
QFrame#cardFrame {
    background-color: #181825;
    border: 2px solid #313244;
    border-radius: 12px;
}

QFrame#videoFrame {
    background-color: #11111b;
    border: 3px solid #45475a;
    border-radius: 12px;
}

QFrame#faceCard {
    background-color: #1e1e2e;
    border: 2px solid #45475a;
    border-radius: 10px;
    padding: 8px;
}

QFrame#faceCard:hover {
    border-color: #89b4fa;
}

/* ComboBox */
QComboBox {
    background-color: #313244;
    color: #cdd6f4;
    border: 2px solid #45475a;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 14px;
}

QComboBox:focus {
    border-color: #89b4fa;
}

QComboBox::drop-down {
    border: none;
    width: 30px;
}

QComboBox QAbstractItemView {
    background-color: #313244;
    color: #cdd6f4;
    selection-background-color: #89b4fa;
    selection-color: #1e1e2e;
    border: 2px solid #45475a;
    border-radius: 6px;
}

/* ToolTip */
QToolTip {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 6px;
}
"""

FACE_CARD_STYLE = """
QFrame#faceCardNew {
    background-color: #1e1e2e;
    border: 3px solid #a6e3a1;
    border-radius: 12px;
}
"""

LOG_COLORS = {
    'new': '#a6e3a1',      # Green
    'sample': '#89b4fa',   # Blue
    'timeout': '#f9e2af',  # Yellow
    'best': '#f5c2e7',     # Pink
    'end': '#cba6f7',      # Purple
    'error': '#f38ba8',    # Red
    'frame': '#6c7086',    # Gray
    'info': '#cdd6f4',     # White
    'success': '#a6e3a1',  # Green
    'warning': '#f9e2af',  # Yellow
    'INFO': '#cdd6f4',     # White (uppercase fallback)
    'NEW': '#a6e3a1',      # Green (uppercase fallback)
    'SAMPLE': '#89b4fa',   # Blue (uppercase fallback)
}
