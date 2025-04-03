import sys
import subprocess
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel,
                             QPushButton, QComboBox, QLineEdit, QMessageBox, QHBoxLayout)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from virtual_keyboard import VirtualKeyboard

class WifiConnector(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Wi-Fi Connector")
        self.setFixedSize(800, 400)
        self.setStyleSheet("background-color: #222; color: white;")
        
        
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        
        title = QLabel("Wi-Fi Connection Manager")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        rowLayout = QHBoxLayout()
        
        self.networksBox = QComboBox()
        rowLayout.addWidget(QLabel("Select Network:"))
        rowLayout.addWidget(self.networksBox)
        
        self.passwordInput = QLineEdit()
        self.passwordInput.setEchoMode(QLineEdit.Password)
        self.passwordInput.setStyleSheet("background-color: #333; color: white; border: 1px solid #555; border-radius: 5px; padding: 5px;")
        rowLayout.addWidget(QLabel("Password:"))
        rowLayout.addWidget(self.passwordInput)
        
        layout.addLayout(rowLayout)
        
        buttonLayout = QHBoxLayout()
        
        self.refreshButton = QPushButton("Refresh")
        self.refreshButton.clicked.connect(self.scan_networks)
        buttonLayout.addWidget(self.refreshButton)
        
        self.connectButton = QPushButton("Connect")
        self.connectButton.clicked.connect(self.connect_to_network)
        buttonLayout.addWidget(self.connectButton)

        self.refreshButton.setStyleSheet("background-color: #555; color: white; border-radius: 5px; padding: 5px;")
        self.connectButton.setStyleSheet("background-color: #555; color: white; border-radius: 5px; padding: 5px;")

        layout.addLayout(buttonLayout)
        
        self.keyboard = VirtualKeyboard(self.passwordInput)
        layout.addWidget(self.keyboard)
        
        self.setLayout(layout)
        self.scan_networks()
    
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
                result = subprocess.check_output(["netsh", "wlan", "show", "network"], encoding="utf-8")
                networks = [line.split(':')[1].strip() for line in result.split('\n') if "SSID" in line]
            else:
                result = subprocess.check_output(["nmcli", "dev", "wifi", "list"], encoding="utf-8")
                networks = [line.split()[0] for line in result.split('\n')[1:] if line]
            return list(set(networks))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to scan networks: {e}")
            return []
    
    def connect_to_network(self):
        ssid = self.networksBox.currentText()
        password = self.passwordInput.text()
        if not ssid:
            QMessageBox.warning(self, "Error", "Please select a network.")
            return
        
        try:
            if sys.platform == "win32":
                cmd = ["netsh", "wlan", "connect", f"name={ssid}"]
                if password:
                    cmd.append(f"key={password}")
            else:
                cmd = ["nmcli", "dev", "wifi", "connect", ssid, "password", password]
            subprocess.run(cmd, check=True)
            QMessageBox.information(self, "Success", f"Connected to {ssid}.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to connect: {e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = WifiConnector()
    window.show()
    sys.exit(app.exec_())
