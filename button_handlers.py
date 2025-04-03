from gpiozero import Button
import os

# GPIO Pin definitions
PIN_MODE = 4
PIN_UP = 17
PIN_DOWN = 27

def set_volume(level):
    capped_level = min(90, max(0, level))
    os.system(f"amixer -D pulse sset Master {capped_level}%")

def volume_up():
    current = get_volume()
    set_volume(min(90, current + 5))

def volume_down():
    current = get_volume()
    set_volume(max(0, current - 5))

def get_volume():
    result = os.popen("amixer -D pulse get Master").read()
    volume = int(result.split('[')[1].split('%')[0])
    return volume

def create_buttons(change_mode_callback):
    button_mode = Button(PIN_MODE, pull_up=True, bounce_time=0.2)
    button_up = Button(PIN_UP, pull_up=True, bounce_time=0.2)
    button_down = Button(PIN_DOWN, pull_up=True, bounce_time=0.2)
    
    button_mode.when_pressed = change_mode_callback
    button_up.when_pressed = volume_up
    button_down.when_pressed = volume_down
    
    return button_mode, button_up, button_down