from picamera2 import Picamera2
import time

# Initialize the camera
picam2 = Picamera2()

# Configure the camera for video recording
video_config = picam2.create_video_configuration(main={"size": (1920, 1080)})
picam2.configure(video_config)

# Start recording
picam2.start()
print("Recording video for 10 seconds...")
picam2.start_recording("/home/pi/video_test.h264")

time.sleep(10)

# Stop recording
picam2.stop_recording()
picam2.stop()
print("Video saved as /home/pi/video_test.h264")
