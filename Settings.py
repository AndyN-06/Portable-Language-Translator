import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QComboBox, QLabel, QPushButton
from PyQt5.QtCore import Qt
from translator_device import TranslatorDevice  # Assuming the device code is in translator_device.py

class TranslatorSettingsUI(QWidget):
    def __init__(self, translator_device):
        super().__init__()

        self.translator_device = translator_device  # The TranslatorDevice object passed to this UI

        # Set up the window
        self.setWindowTitle("Translator Settings")
        self.setGeometry(300, 300, 300, 200)

        # Layout to hold widgets
        layout = QVBoxLayout()

        # Label for the language dropdown
        self.language_label = QLabel("Select Base Language:")
        layout.addWidget(self.language_label)

        # Language dropdown
        self.language_combo = QComboBox(self)
        self.language_combo.addItems(["English", "Spanish", "Korean"])
        layout.addWidget(self.language_combo)

        # Label for the gender dropdown
        self.gender_label = QLabel("Select Voice Gender:")
        layout.addWidget(self.gender_label)

        # Gender dropdown
        self.gender_combo = QComboBox(self)
        self.gender_combo.addItems(["Male", "Female"])
        layout.addWidget(self.gender_combo)

        # Apply button to update settings
        self.apply_button = QPushButton("Apply Settings", self)
        self.apply_button.clicked.connect(self.apply_settings)
        layout.addWidget(self.apply_button)

        # Set the layout
        self.setLayout(layout)

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

# Initialize the TranslatorDevice object
translator_device = TranslatorDevice()

# Create the PyQt5 application
app = QApplication(sys.argv)

# Create the settings window
window = TranslatorSettingsUI(translator_device)
window.show()

# Start the PyQt5 event loop
sys.exit(app.exec_())