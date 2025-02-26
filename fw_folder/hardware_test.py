from speaker import Speaker
from oled import OLED_Display
from mic import AudioRecorder
import time

# init variabless / hardware
oled_display = OLED_Display()
recorder = AudioRecorder(duration=5, sample_rate=44100, file_name="output.wav")
speaker = Speaker()

# Start test
oled_display.write_text("Beginning of Test", 10, 20)
time.sleep(1)

# Begin Recording Audio
oled_display.clear_display()
oled_display.write_text("Recording Audio...", 10, 20)
recorder.record()
recorder.save_to_file()

# Output done messages
oled_display.clear_display()
oled_display.write_text("Audio Done Recording", 10, 20)
time.sleep(0.5)
oled_display.clear_display()
oled_display.write_text("Saved to Audio File", 10, 20)

# start outputting audio to speaker
oled_display.clear_display()
oled_display.write_text("Outputting Audio...", 10, 20)
speaker.play_audio("output.wav") 
