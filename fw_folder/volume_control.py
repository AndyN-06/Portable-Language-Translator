import os
import time

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
    # Extract the volume percentage from the output
    volume = int(result.split('[')[1].split('%')[0])
    return volume

def adjust_volume():
    """Decrease volume until 0, then increase"""
    increasing = False
    while True:
        current_volume = get_volume()
        
        if increasing:
            if current_volume < 100:
                increase_volume(5)
            if current_volume == 100:
                increasing = False
        else:
            if current_volume > 0:
                decrease_volume(5)
            if current_volume == 0:
                increasing = True
        
        # Wait 5 seconds before adjusting volume again
        time.sleep(5)

if __name__ == "__main__":
    adjust_volume()

