"""UI Package for YOLOFace Tracker Application"""

# Lazy imports to avoid circular import issues with PyInstaller
def __getattr__(name):
    if name == 'MainWindow':
        from .main_window import MainWindow
        return MainWindow
    elif name == 'FaceCard':
        from .face_card import FaceCard
        return FaceCard
    elif name == 'FaceCardCompact':
        from .face_card import FaceCardCompact
        return FaceCardCompact
    elif name == 'VideoDisplay':
        from .video_display import VideoDisplay
        return VideoDisplay
    elif name == 'ImageDisplay':
        from .video_display import ImageDisplay
        return ImageDisplay
    elif name == 'MAIN_STYLE':
        from .styles import MAIN_STYLE
        return MAIN_STYLE
    elif name == 'LOG_COLORS':
        from .styles import LOG_COLORS
        return LOG_COLORS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['MainWindow', 'FaceCard', 'FaceCardCompact', 'VideoDisplay', 'ImageDisplay', 'MAIN_STYLE', 'LOG_COLORS']