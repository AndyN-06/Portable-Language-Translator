import sys
import os
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QLabel, QTabWidget, QWidget, QMessageBox)

# Global volume variable
current_volume = 50  # Default to 50% volume

# Update the global volume and show a message box
def update_volume(new_volume):
    global current_volume
    current_volume = new_volume
    # Show a pop-up with the current volume level
    QMessageBox.information(None, "Volume Changed", f"Current volume: {current_volume}%")

# Set volume function
def set_volume(level):
    # Ensure level doesn't exceed 90%
    capped_level = min(90, max(0, level))
    os.system(f"amixer -D pulse sset Master {capped_level}%")
    update_volume(capped_level)  # Update the volume and show the pop-up

# Get volume function
def get_volume():
    result = os.popen("amixer -D pulse get Master").read()
    volume = int(result.split('[')[1].split('%')[0])
    update_volume(volume)  # Update the volume when fetched
    return volume

# Increase volume function
def increase_volume(step=5):
    current = get_volume()
    # Calculate new volume but don't exceed 90%
    new_volume = min(90, current + step)
    set_volume(new_volume)

    return get_volume()

# Decrease volume function
def decrease_volume(step=5):
    current = get_volume()
    # Ensure volume doesn't go below 0
    new_volume = max(0, current - step)
    set_volume(new_volume)

    return get_volume()

# Main window with volume control loop
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Volume Control Test")
        self.setGeometry(100, 100, 800, 500)
        
        self.initUI()
    
    def initUI(self):
        layout = QVBoxLayout()
        label = QLabel("Testing Volume Control Loop")
        layout.addWidget(label)
        
        # Set up the timer for the volume control loop
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.volume_control_loop)
        self.timer.start(2000)  # Adjust volume every 2 seconds
        
        # Start with setting volume
        set_volume(50)  # Start at 50% volume

        # Set the layout for the main window
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def volume_control_loop(self):
        # Test the volume control by alternating increase and decrease
        current = get_volume()
        if current < 90:
            increase_volume(5)
        else:
            decrease_volume(5)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
