import sys
import subprocess
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QTabWidget, QFrame,
                             QPushButton, QComboBox, QLineEdit, QMessageBox, QHBoxLayout)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from virtual_keyboard import VirtualKeyboard

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("PyQt Tab Example")
        self.setGeometry(100, 100, 800, 500)
        
        self.initUI()
    
    def initUI(self):
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #aaa; background: #ddd; }
            QTabBar::tab { padding: 6px; font-size: 10px; background: #eee; color: black; border: 1px solid #aaa; }
            QTabBar::tab:selected { background: #ccc; }
        """)
        
        # Create Tabs
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        
        # Add tabs to the QTabWidget
        self.tabs.addTab(self.tab1, "Home")
        self.tabs.addTab(self.tab2, "Wi-Fi")
        self.tabs.addTab(self.tab3, "About")
        
        # Set up layouts for each tab
        self.setupTab1()
        self.setupTab2()
        self.setupTab3()
        
        self.setCentralWidget(self.tabs)
    
    def setupTab1(self):
        layout = QVBoxLayout()
        label = QLabel("Welcome to the Home Tab")
        label.setFont(QFont("Arial", 16))
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("color: black;")
        layout.addWidget(label)
        self.tab1.setStyleSheet("background-color: #fff;")
        self.tab1.setLayout(layout)
    
    def setupTab2(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        
        title = QLabel("Wi-Fi Connection Manager")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: black;")
        layout.addWidget(title)
        
        rowLayout = QHBoxLayout()
        
        # Change QComboBox to QLineEdit
        self.networksInput = QLineEdit()
        self.networksInput.setPlaceholderText("Enter network name")
        self.networksInput.setStyleSheet("background-color: #fff; color: black; border: 1px solid #aaa;")
        rowLayout.addWidget(QLabel("Network:", self))
        rowLayout.addWidget(self.networksInput)
        
        self.passwordInput = QLineEdit()
        self.passwordInput.setEchoMode(QLineEdit.Password)
        self.passwordInput.setStyleSheet("background-color: #fff; color: black; border: 1px solid #aaa; border-radius: 5px; padding: 5px;")
        rowLayout.addWidget(QLabel("Password:", self))
        rowLayout.addWidget(self.passwordInput)
        
        layout.addLayout(rowLayout)
        
        buttonLayout = QHBoxLayout()
        
        
        self.connectButton = QPushButton("Connect")
        self.connectButton.setStyleSheet("background-color: #ccc; color: black; border-radius: 5px; padding: 5px;")
        self.connectButton.clicked.connect(self.connect_to_network)
        buttonLayout.addWidget(self.connectButton)
        
        layout.addLayout(buttonLayout)
        
        self.keyboard = VirtualKeyboard(self.passwordInput)
        layout.addWidget(self.keyboard)
        
        self.tab2.setStyleSheet("background-color: #fff;")
        self.tab2.setLayout(layout)
    
    def setupTab3(self):
        layout = QVBoxLayout()
        label = QLabel("About this application")
        label.setFont(QFont("Arial", 16))
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("color: black;")
        layout.addWidget(label)
        self.tab3.setStyleSheet("background-color: #fff;")
        self.tab3.setLayout(layout)
    
    def connect_to_network(self):
        ssid = self.networksInput.text()  # Fetch the selected SSID from the ComboBox
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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
