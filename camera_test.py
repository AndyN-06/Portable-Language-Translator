from picamera2 import Picamera2
import time

def test_picam():
    # Initialize the camera
    picam2 = Picamera2()
    
    # Configure the camera
    preview_config = picam2.create_preview_configuration()
    picam2.configure(preview_config)
    
    # Start the camera
    picam2.start()
    
    # Wait for camera to initialize
    time.sleep(2)
    
    # Get current camera configuration
    print("Camera Configuration:")
    print(f"Sensor Mode: {picam2.camera_properties}")
    print(f"Camera Configuration: {picam2.camera_configuration}")
    
    # Capture a test image
    print("\nCapturing test image...")
    picam2.capture_file("test_image.jpg")
    print("Test image saved as 'test_image.jpg'")
    
    # Stop the camera
    picam2.stop()

if __name__ == "__main__":
    try:
        test_picam()
    except Exception as e:
        print(f"Error: {str(e)}")