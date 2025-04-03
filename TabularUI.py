import sys
import subprocess
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QTabWidget, QFrame,
                             QPushButton, QComboBox, QLineEdit, QMessageBox, QHBoxLayout, QTextEdit, QSizePolicy, QProgressBar)
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt, QFileSystemWatcher, QTime
from virtual_keyboard import VirtualKeyboard
import re
import cv2
import os
from translator_device import TranslatorDevice  # Assuming the device code is in translator_device.py

def get_volume():
    result = os.popen("amixer -D pulse get Master").read()
    volume = int(result.split('[')[1].split('%')[0])
    return volume

class MainWindow(QMainWindow):
    def __init__(self, filepath, translator_device):
        super().__init__()

        self.file_path = filepath
        self.translator_device = translator_device

        self.setWindowTitle("PyQt Tab Example")
        self.setGeometry(100, 100, 800, 500)
        
        self.initUI()
    
    def initUI(self):
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""QTabWidget::pane { border: 1px solid #aaa; background: #ddd; }
                                  QTabBar::tab { padding: 6px; font-size: 10px; background: #eee; color: black; border: 1px solid #aaa; }
                                  QTabBar::tab:selected { background: #ccc; }""")
        
        # Create Tabs
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        
        # Add tabs to the QTabWidget
        self.tabs.addTab(self.tab1, "Translation")
        self.tabs.addTab(self.tab2, "Wi-Fi")
        self.tabs.addTab(self.tab3, "Settings")
        
        # Set up layouts for each tab
        self.setupTab1()
        self.setupTab2()
        self.setupTab3()
        
        self.setCentralWidget(self.tabs)
    
    def setupTab1(self):
        main_layout = QVBoxLayout()  # Use QVBoxLayout for vertical stacking

        # Create video label and text edit widget
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedSize(640, 400)
        
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)

        # Create the status label that will be updated
        self.status_label = QLabel(self)
        self.status_label.setAlignment(Qt.AlignLeft)  # Align to the left
        self.status_label.setStyleSheet("""
        font-size: 12pt;
        color: white;
        background-color: green;
        padding: 5px;
        border-radius: 5px;
        """)  # Style it with green background, white text, and rounded corners

        # Set fixed size for the status label (optional)
        self.status_label.setFixedSize(200, 30)  # Small size for the label

        # Initially, show camera view and text edit
        main_layout.addWidget(self.status_label)  # Add status label at the top of the layout
        main_layout.addWidget(self.video_label)
        main_layout.addWidget(self.text_edit)
        self.tab1.setLayout(main_layout)

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

        # Set up a timer to update the status label periodically
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(1000)  # Update every 1000ms (1 second)

    def update_status(self):
        # This function updates the status label's text
        from shared import mode
        self.status_label.setText("mode: " + mode)
        if mode == "SPEECH":
            self.status_label.setStyleSheet("""
            font-size: 12pt;
            color: black;
            background-color: green;
            padding: 5px;
            border-radius: 5px;
            """)
        else:
            self.status_label.setStyleSheet("""
            font-size: 12pt;
            color: black;
            background-color: red;
            padding: 5px;
            border-radius: 5px;
            """)



    def setupTab2(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        
        title = QLabel("Wi-Fi Connection Manager")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: black;")
        layout.addWidget(title)
        
        rowLayout = QHBoxLayout()
        
        self.networksBox = QComboBox()
        self.networksBox.setEditable(True)  # Allow keyboard input
        self.networksBox.setStyleSheet("background-color: #fff; color: black; border: 1px solid #aaa;")
        rowLayout.addWidget(QLabel("Select Network:", self))
        rowLayout.addWidget(self.networksBox)
        
        self.passwordInput = QLineEdit()
        self.passwordInput.setEchoMode(QLineEdit.Password)
        self.passwordInput.setStyleSheet("background-color: #fff; color: black; border: 1px solid #aaa; border-radius: 5px; padding: 5px;")
        rowLayout.addWidget(QLabel("Password:", self))
        rowLayout.addWidget(self.passwordInput)
        
        layout.addLayout(rowLayout)
        
        buttonLayout = QHBoxLayout()
        
        self.refreshButton = QPushButton("Refresh")
        self.refreshButton.setStyleSheet("background-color: #ccc; color: black; border-radius: 5px; padding: 5px;")
        self.refreshButton.clicked.connect(self.scan_networks)
        buttonLayout.addWidget(self.refreshButton)
        
        self.connectButton = QPushButton("Connect")
        self.connectButton.setStyleSheet("background-color: #ccc; color: black; border-radius: 5px; padding: 5px;")
        self.connectButton.clicked.connect(self.connect_to_network)
        buttonLayout.addWidget(self.connectButton)
        
        layout.addLayout(buttonLayout)
        
        self.keyboard = VirtualKeyboard(self.passwordInput)
        layout.addWidget(self.keyboard)
        
        self.tab2.setStyleSheet("background-color: #fff;")
        self.tab2.setLayout(layout)
        self.scan_networks()

    def setupTab3(self):
        layout = QVBoxLayout()
        layout.setSpacing(8)  # Reduced spacing between elements
        layout.setContentsMargins(10, 10, 10, 10)  # Minimized margins

        # Volume progress bar
        self.volume_bar = QProgressBar(self)
        self.volume_bar.setRange(0, 90)  # Volume range 0-90%
        self.volume_bar.setValue(get_volume())  # Initial volume level
        layout.addWidget(self.volume_bar)

        # Container for centering
        container = QWidget()
        container_layout = QVBoxLayout()
        container_layout.setAlignment(Qt.AlignCenter)

        # Grid Layout for compact arrangement
        grid_layout = QHBoxLayout()
        grid_layout.setAlignment(Qt.AlignCenter)

        # Language selection
        self.language_label = QLabel("Language:")
        self.language_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        
        self.language_combo = QComboBox(self)
        self.language_combo.addItems(["English", "Spanish", "Korean"])
        self.language_combo.setFixedWidth(100)  # Smaller dropdown width

        # Gender selection
        self.gender_label = QLabel("Voice:")
        self.gender_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.gender_combo = QComboBox(self)
        self.gender_combo.addItems(["Male", "Female"])
        self.gender_combo.setFixedWidth(100)  # Smaller dropdown width

        # Arrange dropdowns and labels in a row
        grid_layout.addWidget(self.language_label)
        grid_layout.addWidget(self.language_combo)
        grid_layout.addWidget(self.gender_label)
        grid_layout.addWidget(self.gender_combo)

        # Apply button
        self.apply_button = QPushButton("Apply")
        self.apply_button.setFixedSize(100, 40)  # Bigger button
        self.apply_button.setStyleSheet("font-size: 12pt;")  # Larger text
        self.apply_button.clicked.connect(self.apply_settings)

        # Reduce spacing between dropdowns and button
        container_layout.addLayout(grid_layout)
        container_layout.addSpacing(20)  # Small spacing
        container_layout.addWidget(self.apply_button, alignment=Qt.AlignCenter)

        # Center content in the screen
        container.setLayout(container_layout)
        layout.addWidget(container, alignment=Qt.AlignCenter)

        # Apply layout to Tab 3
        self.tab3.setStyleSheet("background-color: #fff;")
        self.tab3.setLayout(layout)

        # Timer to periodically update the volume level bar
        self.volume_timer = QTimer(self)
        self.volume_timer.timeout.connect(self.update_volume_bar)
        self.volume_timer.start(100)  # Update every second



    def scan_networks(self):
        self.networksBox.clear()
        networks = self.get_available_networks()
        if networks:
            self.networksBox.addItems(networks)
        else:
            QMessageBox.warning(self, "Error", "No networks found.")
    
    def get_available_networks(self):
        try:
            if sys.platform == "win32":
                result = subprocess.check_output(["netsh", "wlan", "show", "network"], encoding="utf-8", errors="ignore")
                print("Raw netsh output:\n", result)  # Debugging

                networks = []
                for line in result.split('\n'):
                    if "SSID" in line and ":" in line:
                        ssid = line.split(':', 1)[1].strip()
                        ssid = ssid.replace("?T", "'")  # Fix apostrophe
                        networks.append(ssid)

                return list(set(networks))
            else:
                result = subprocess.check_output(["nmcli", "dev", "wifi", "list"], encoding="utf-8", errors="ignore")
                print("Raw nmcli output:\n", result)  # Debugging
                matches = re.findall(r'(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\s+(.+?)\s+Infra', result)
                networks = set(ssid.strip() for ssid in matches)

            return list(set(networks))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to scan networks: {e}")
            return []
 
    def connect_to_network(self):
        ssid = self.networksBox.currentText()  # Fetch the selected SSID from the ComboBox
        password = self.passwordInput.text().strip()

        if not ssid:
            QMessageBox.warning(self, "Error", "Please select a network.")
            return

        try:
            if sys.platform == "win32":  # If Windows
                # Use netsh command to connect to the network
                cmd = f'netsh wlan connect name="{ssid}"'
                print("Executing (Windows):", cmd)  # Debugging
                subprocess.run(cmd, shell=True, check=True)

            elif sys.platform != "win32":  # If Raspberry Pi (Linux)
                # Use nmcli command to connect to the selected network with the password
                cmd = f'nmcli dev wifi connect "{ssid}" password "{password}"'
                print("Executing (Linux):", cmd)  # Debugging
                subprocess.run(cmd, shell=True, check=True)

            else:
                print("This script is only for Windows or Linux (Raspberry Pi).")
                return

            print(f"Successfully connected to {ssid}.")
            QMessageBox.information(self, "Success", f"Successfully connected to {ssid}.")

        except subprocess.CalledProcessError as e:
            QMessageBox.critical(self, "Error", f"Failed to connect: {e}")

    def update_camera(self):
        from shared import latest_frame, ui_mode
        if ui_mode == "CAMERA" and latest_frame is not None:
            frame = latest_frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            # Change scaling flag from Qt.KeepAspectRatio to Qt.IgnoreAspectRatio 
            # (or use KeepAspectRatioByExpanding) so that the image fills the label.
            pixmap = QPixmap.fromImage(qt_image).scaled(self.video_label.width(),
                                                        self.video_label.height(),
                                                        Qt.KeepAspectRatio, # Use KeepAspectRatio
                                                        Qt.FastTransformation) # CHANGED
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

    def update_volume_bar(self):
        # Update the volume progress bar with the current volume level
        volume = get_volume()
        self.volume_bar.setValue(volume)  # Update the progress bar
    
    def apply_settings(self):
        """Apply the selected settings to the translator device."""
        selected_language = self.language_combo.currentText()
        selected_gender = self.gender_combo.currentText()

        # Map the dropdown values to the respective language and gender
        language_mapping = {
            "English": "en-US",
            "Spanish": "es-US",
            "Korean": "ko-KR"
        }

        gender_mapping = {
            "Male": "MALE",
            "Female": "FEMALE"
        }

        base_language = language_mapping.get(selected_language, "en-US")
        gender = gender_mapping.get(selected_gender, "MALE")

        # Update the settings of the translator device
        self.translator_device.set_settings(base_language, gender)
        print(f"Settings updated: Language - {base_language}, Gender - {gender}")

if __name__ == "__main__":
    file_path = "als_speech_audio_transcription.txt"  # This file is updated by the ASL processing thread
    with open(file_path, 'w') as file:
        pass  # clear the file contents

    translator_device = TranslatorDevice()
    app = QApplication(sys.argv)
    window = MainWindow(file_path, translator_device)
    window.show()
    sys.exit(app.exec_())