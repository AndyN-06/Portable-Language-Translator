from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout, QGridLayout, QFrame
from PyQt5.QtCore import Qt

class VirtualKeyboard(QWidget):
    def __init__(self, target_field):
        super().__init__()
        self.target_field = target_field
        self.shift = False
        self.caps_lock = False
        self.buttons = {}
        self.initUI()

    def initUI(self):
        self.setStyleSheet("background-color: #222; border-radius: 10px;")

        layout = QVBoxLayout()
        self.keys = [
            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
            ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
            ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', '←'],
            ['↑', 'z', 'x', 'c', 'v', 'b', 'n', 'm', '␣']
        ]

        keyboard_frame = QFrame()
        keyboard_frame.setStyleSheet("background-color: #333; border-radius: 10px; padding: 10px;")
        grid_layout = QGridLayout()

        for row_idx, row in enumerate(self.keys):
            for col_idx, key in enumerate(row):
                btn = QPushButton(key.upper() if self.caps_lock else key)
                btn.setFixedSize(50, 50)
                btn.setStyleSheet(self.get_button_style(key))
                btn.clicked.connect(lambda checked, k=key: self.key_pressed(k))
                grid_layout.addWidget(btn, row_idx, col_idx)
                self.buttons[key] = btn

        keyboard_frame.setLayout(grid_layout)
        layout.addWidget(keyboard_frame)
        self.setLayout(layout)

    def key_pressed(self, key):
        shift_symbols = {
            '1': '!', '2': '@', '3': '#', '4': '$', '5': '%',
            '6': '^', '7': '&&', '8': '*', '9': '(', '0': ')'
        }

        if key == "␣":
            self.target_field.setText(self.target_field.text() + " ")
        elif key == "←":  # Backspace
            self.target_field.setText(self.target_field.text()[:-1])
        elif key == "↑":  # Shift / Caps Lock
            if self.caps_lock:
                self.caps_lock = False
            elif self.shift:
                self.caps_lock = True
                self.shift = False
            else:
                self.shift = True
            self.update_keys()
        else:
            # Handle numbers and their shift symbols
            if key in shift_symbols and self.shift:
                self.target_field.setText(self.target_field.text() + shift_symbols[key])
            else:
                if self.shift or self.caps_lock:
                    self.target_field.setText(self.target_field.text() + key.upper())
                else:
                    self.target_field.setText(self.target_field.text() + key.lower())

            self.shift = False  # Shift only applies to the next key press
            self.update_keys()


    def update_keys(self):
        shift_symbols = {
            '1': '!', '2': '@', '3': '#', '4': '$', '5': '%',
            '6': '^', '7': '&&', '8': '*', '9': '(', '0': ')'
        }

        for key, btn in self.buttons.items():
            if key.isalpha():
                btn.setText(key.upper() if self.shift or self.caps_lock else key.lower())
            elif key in shift_symbols:
                btn.setText(shift_symbols[key] if self.shift else key)

        # Update Shift key appearance
        shift_btn = self.buttons.get("↑")
        if shift_btn:
            if self.caps_lock:
                shift_btn.setStyleSheet(self.get_button_style("↑", caps_lock=True))
            elif self.shift:
                shift_btn.setStyleSheet(self.get_button_style("↑", shift=True))
            else:
                shift_btn.setStyleSheet(self.get_button_style("↑"))


    def get_button_style(self, key, shift=False, caps_lock=False):
        base_style = """
            QPushButton {
                font-size: 20px; color: white; background-color: #444;
                border-radius: 10px; border: 1px solid #555;
            }
            QPushButton:pressed {
                background-color: #666;
            }
        """
        if key == "↑":
            if caps_lock:
                return base_style + "QPushButton { background-color: #ffcc00; text-decoration: underline; }"
            elif shift:
                return base_style + "QPushButton { background-color: #ffcc00; }"
        return base_style
