import sys
import cv2
import os
from PyQt5.QtCore import QTimer, Qt, QFileSystemWatcher
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QTextEdit, QHBoxLayout
# from everything import latest_frame

class CameraTextViewer(QWidget):
    def __init__(self, file_path):
        super().__init__()

        self.file_path = file_path

        # Set up the UI
        self.setWindowTitle("Camera & Text Viewer")
        self.showFullScreen()  # Make the window full screen

        # Layout
        main_layout = QHBoxLayout()  # Horizontal layout for side-by-side view

        # Left side: Camera Feed
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)

        # Right side: Text File Viewer
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)  # Make text read-only

        # Add widgets to layout
        main_layout.addWidget(self.video_label, 2)  # Camera takes 2/3 of space
        main_layout.addWidget(self.text_edit, 1)  # Text takes 1/3 of space
        self.setLayout(main_layout)

        # Open the webcam
        # self.cap = cv2.VideoCapture(0)

        # Timer for updating the camera feed
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_camera)
        self.camera_timer.start(30)  # Update every 30ms (~33 FPS)

        # File watcher for detecting text file changes
        self.file_watcher = QFileSystemWatcher()
        self.file_watcher.addPath(self.file_path)
        self.file_watcher.fileChanged.connect(self.load_text)

        # Load initial text
        self.load_text()

    # def update_camera(self):
    #     """Capture a frame from the webcam and display it."""
    #     ret, frame = self.cap.read()
    #     if ret:
    #         # Convert frame from BGR (OpenCV default) to RGB (Qt format)
    #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #         # Convert to QImage
    #         h, w, ch = frame.shape
    #         bytes_per_line = ch * w
    #         qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

    #         # Scale image to fit QLabel size
    #         pixmap = QPixmap.fromImage(qt_image).scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio)
    #         self.video_label.setPixmap(pixmap)
    
    def update_camera(self, latest_frame):
        """Capture a frame from the webcam and display it."""
        # from everything import latest_frame  # Ensure the latest_frame is imported dynamically
        if latest_frame is not None:
            frame = latest_frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image).scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio)
            self.video_label.setPixmap(pixmap)

    def load_text(self):
        """Load the text file content into the QTextEdit widget."""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as file:
                    content = file.read()
                    self.text_edit.setText(content)
            except Exception as e:
                self.text_edit.setText(f"Error loading file: {e}")
        else:
            self.text_edit.setText("File not found.")

    # def closeEvent(self, event):
    #     """Override close event to release the camera."""
    #     self.cap.release()
    #     event.accept()

# Run the application
# if __name__ == "__main__":
#     file_path = "text.txt"  # Change this to your actual file path
#     app = QApplication(sys.argv)
#     window = CameraTextViewer(file_path)
#     window.show()
#     sys.exit(app.exec_())

