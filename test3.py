import sys
import os
import subprocess
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QSlider, QProgressBar
from PyQt5.QtCore import Qt, QTimer


def set_volume(level):
    # Ensure level doesn't exceed 90%
    capped_level = min(90, max(0, level))
    os.system(f"amixer -D pulse sset Master {capped_level}%")


def increase_volume(step=5):
    current = get_volume()
    # Calculate new volume but don't exceed 90%
    new_volume = min(90, current + step)
    set_volume(new_volume)


def decrease_volume(step=5):
    current = get_volume()
    # Ensure volume doesn't go below 0
    new_volume = max(0, current - step)
    set_volume(new_volume)


def get_volume():
    result = os.popen("amixer -D pulse get Master").read()
    volume = int(result.split('[')[1].split('%')[0])
    return volume


class VolumeControlApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Volume Control")
        self.setGeometry(200, 200, 300, 250)

        # Layout
        self.layout = QVBoxLayout()

        # Volume control slider
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 90)
        self.volume_slider.setValue(get_volume())
        self.volume_slider.valueChanged.connect(self.update_volume_from_slider)
        self.layout.addWidget(self.volume_slider)

        # Volume control buttons
        self.increase_button = QPushButton("Increase Volume")
        self.increase_button.clicked.connect(lambda: increase_volume(5))
        self.layout.addWidget(self.increase_button)

        self.decrease_button = QPushButton("Decrease Volume")
        self.decrease_button.clicked.connect(lambda: decrease_volume(5))
        self.layout.addWidget(self.decrease_button)

        # Volume progress bar
        self.volume_bar = QProgressBar(self)
        self.volume_bar.setRange(0, 90)
        self.volume_bar.setValue(get_volume())  # Initial volume level
        self.layout.addWidget(self.volume_bar)

        # Set the layout of the window
        self.setLayout(self.layout)

        # Timer to periodically update the volume level bar
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_volume_bar)
        self.timer.start(1000)  # Update every second

    def update_volume_from_slider(self):
        # Get value from the slider and set the volume
        level = self.volume_slider.value()
        set_volume(level)

    def update_volume_bar(self):
        # Update the volume progress bar with the current volume level
        volume = get_volume()
        self.volume_bar.setValue(volume)  # Update the progress bar

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VolumeControlApp()
    window.show()
    sys.exit(app.exec_())
