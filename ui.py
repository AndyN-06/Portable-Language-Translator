# UI.py
import sys
import cv2
import os
from PyQt5.QtCore import QTimer, Qt, QFileSystemWatcher
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QTextEdit, QHBoxLayout
# from shared import latest_frame

class CameraTextViewer(QWidget):
    def __init__(self, file_path):
        super().__init__()

        self.file_path = file_path
        self.setWindowTitle("Camera & Text Viewer")
        self.showFullScreen()

        main_layout = QHBoxLayout()
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)

        # Initially, show camera view only
        main_layout.addWidget(self.video_label)
        main_layout.addWidget(self.text_edit)
        self.setLayout(main_layout)

        # Set up timers
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_camera)
        self.camera_timer.start(30)  # Update every 30ms

        self.mode_timer = QTimer()
        self.mode_timer.timeout.connect(self.update_ui_mode)
        self.mode_timer.start(500)  # Check UI mode every 500ms

        self.file_watcher = QFileSystemWatcher()
        self.file_watcher.addPath(self.file_path)
        self.file_watcher.fileChanged.connect(self.load_text)

        self.text_edit.setStyleSheet("font-size: 50pt;")

        self.load_text()

    def update_camera(self):
        """Display the latest camera frame if in CAMERA mode."""
        from shared import latest_frame, ui_mode
        if ui_mode == "CAMERA" and latest_frame is not None:
            frame = latest_frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image).scaled(self.video_label.width(),
                                                        self.video_label.height(),
                                                        Qt.KeepAspectRatio)
            self.video_label.setPixmap(pixmap)
        else:
            self.video_label.clear()

    def update_ui_mode(self):
        """Switch between camera view and text view based on the shared ui_mode variable."""
        from shared import ui_mode
        if ui_mode == "CAMERA":
            self.video_label.show()
            self.text_edit.hide()
        elif ui_mode == "TEXT":
            self.video_label.hide()
            self.text_edit.show()

    def load_text(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as file:
                    content = file.read()
                    self.text_edit.setText(content)
            except Exception as e:
                self.text_edit.setText(f"Error loading file: {e}")
        else:
            self.text_edit.setText("File not found.")
