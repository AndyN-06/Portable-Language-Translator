import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, 
                            QTabWidget, QPushButton, QComboBox, QLineEdit, QHBoxLayout, 
                            QTextEdit, QSizePolicy, QProgressBar)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Portable Language Translator - Preview")
        self.setGeometry(100, 100, 800, 500)
        self.initUI()
    
    def initUI(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # Create tabs
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #aaa; background: #ddd; }
            QTabBar::tab { padding: 6px; font-size: 10px; background: #eee; color: black; border: 1px solid #aaa; }
            QTabBar::tab:selected { background: #ccc; }
        """)
        
        # Status label
        self.status_label = QLabel("Mode: SPEECH")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            font-size: 12pt;
            color: black;
            background-color: green;
            padding: 5px;
            border-radius: 5px;
            margin-left: 10px;
        """)
        self.status_label.setFixedSize(120, 30)
        self.tabs.setCornerWidget(self.status_label, Qt.TopRightCorner)
        
        # Create and add tabs
        self.tab1 = self.create_translation_tab()
        self.tab2 = self.create_wifi_tab()
        self.tab3 = self.create_settings_tab()
        
        self.tabs.addTab(self.tab1, "Translation")
        self.tabs.addTab(self.tab2, "Wi-Fi")
        self.tabs.addTab(self.tab3, "Settings")
        
        main_layout.addWidget(self.tabs)
        self.setCentralWidget(main_widget)

    def create_translation_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Video preview area (gray placeholder)
        video_placeholder = QLabel()
        video_placeholder.setStyleSheet("background-color: #888;")
        video_placeholder.setFixedSize(640, 400)
        
        # Text area
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setStyleSheet("font-size: 20pt;")
        text_edit.setText("Translation text will appear here")
        
        layout.addWidget(video_placeholder)
        layout.addWidget(text_edit)
        tab.setLayout(layout)
        return tab

    def create_wifi_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        title = QLabel("Wi-Fi Connection Manager")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        
        network_layout = QHBoxLayout()
        network_layout.addWidget(QLabel("Select Network:"))
        network_combo = QComboBox()
        network_combo.addItems(["Network 1", "Network 2", "Network 3"])
        network_layout.addWidget(network_combo)
        
        password_layout = QHBoxLayout()
        password_layout.addWidget(QLabel("Password:"))
        password_input = QLineEdit()
        password_input.setEchoMode(QLineEdit.Password)
        password_layout.addWidget(password_input)
        
        button_layout = QHBoxLayout()
        refresh_btn = QPushButton("Refresh")
        connect_btn = QPushButton("Connect")
        button_layout.addWidget(refresh_btn)
        button_layout.addWidget(connect_btn)
        
        layout.addWidget(title)
        layout.addLayout(network_layout)
        layout.addLayout(password_layout)
        layout.addLayout(button_layout)
        tab.setLayout(layout)
        return tab

    def create_settings_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        volume_bar = QProgressBar()
        volume_bar.setRange(0, 100)
        volume_bar.setValue(50)
        
        settings_layout = QHBoxLayout()
        
        language_layout = QHBoxLayout()
        language_layout.addWidget(QLabel("Language:"))
        language_combo = QComboBox()
        language_combo.addItems(["English", "Spanish", "Korean"])
        language_combo.setFixedWidth(100)
        language_layout.addWidget(language_combo)
        
        voice_layout = QHBoxLayout()
        voice_layout.addWidget(QLabel("Voice:"))
        voice_combo = QComboBox()
        voice_combo.addItems(["Male", "Female"])
        voice_combo.setFixedWidth(100)
        voice_layout.addWidget(voice_combo)
        
        settings_layout.addLayout(language_layout)
        settings_layout.addLayout(voice_layout)
        
        apply_btn = QPushButton("Apply")
        apply_btn.setFixedSize(100, 40)
        
        layout.addWidget(volume_bar)
        layout.addLayout(settings_layout)
        layout.addWidget(apply_btn, alignment=Qt.AlignCenter)
        tab.setLayout(layout)
        return tab

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())