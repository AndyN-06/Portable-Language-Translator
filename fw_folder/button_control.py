import os
import time
from gpiozero import Button

# Volume control functions
def set_volume(level):
    """Set volume level (0-100%)"""
    os.system(f"amixer -D pulse sset Master {level}%")

def increase_volume(step=5):
    """Increase volume by a step"""
    os.system(f"amixer -D pulse sset Master {step}%+")

def decrease_volume(step=5):
    """Decrease volume by a step"""
    os.system(f"amixer -D pulse sset Master {step}%-")

def get_volume():
    """Get current volume level"""
    result = os.popen("amixer -D pulse get Master").read()
    volume = int(result.split('[')[1].split('%')[0])
    return volume

# GPIO pin setup using gpiozero
PIN_1 = 4
PIN_2 = 17
PIN_3 = 27

mode = 0

def change_mode():
    global mode
    mode = 1 if mode == 0 else 0
    print("Toggled mode:", mode)

def volume_up():
    print("Increased Volume")
    increase_volume()

def volume_down():
    print("Decreased Volume")
    decrease_volume()

# Initialize buttons with pull-up resistors
button_mode = Button(PIN_1, pull_up=True, bounce_time=0.2)
button_up = Button(PIN_2, pull_up=True, bounce_time=0.2)
button_down = Button(PIN_3, pull_up=True, bounce_time=0.2)

# Attach event listeners
button_mode.when_pressed = change_mode
button_up.when_pressed = volume_up
button_down.when_pressed = volume_down

try:
    while True:
        time.sleep(1)  # Keep the script running
except KeyboardInterrupt:
    print("Program exited")

