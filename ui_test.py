import sys
import cv2
import os
import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QTextEdit, QHBoxLayout

class CameraTextViewer(QWidget):
    def __init__(self, file_path):
        super().__init__()

        self.file_path = file_path
        self.setWindowTitle("Camera & Text Viewer")
        self.resize(1200, 600)

        main_layout = QHBoxLayout()
        
        # Video Label
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedSize(600, 600)

        # Text Viewer
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setStyleSheet("font-size: 20pt;")
        self.text_edit.setFixedSize(600, 600)

        main_layout.addWidget(self.video_label)
        main_layout.addWidget(self.text_edit)
        self.setLayout(main_layout)

        # Set up camera
        self.cap = cv2.VideoCapture(0)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30ms

        self.load_text()

    def update_frame(self):
        """Capture and display frames from the camera."""
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Black placeholder
            cv2.putText(frame, "No Camera", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image).scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio)
        self.video_label.setPixmap(pixmap)

    def load_text(self):
        """Load text from the specified file."""
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w", encoding="utf-8") as file:
                file.write("This is a sample text file.\nModify this text to see updates in the UI.")

        with open(self.file_path, "r", encoding="utf-8") as file:
            content = file.read()
            self.text_edit.setText(content)

    def closeEvent(self, event):
        """Release camera resource on close."""
        if self.cap.isOpened():
            self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    file_path = "sample_text.txt"
    viewer = CameraTextViewer(file_path)
    viewer.show()
    sys.exit(app.exec_())
