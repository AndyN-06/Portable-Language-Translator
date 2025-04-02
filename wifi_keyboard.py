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
                result = subprocess.check_output(["nmcli", "dev", "wifi", "list"], encoding="utf-8")
                print("Raw nmcli output:\n", result)  # Debugging
                networks = [line.split()[0] for line in result.split('\n')[1:] if line]
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

        if not password:
            QMessageBox.warning(self, "Error", "Please enter a password.")
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



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = WifiConnector()
    window.show()
    sys.exit(app.exec_())