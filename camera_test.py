import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

standard_resolutions = [
    (160, 120),  # QQVGA
    (320, 240),  # QVGA
    (480, 360),  # nHD
    (640, 480),  # VGA
    (800, 600),  # SVGA
    (1024, 768), # XGA
    (1280, 720), # HD
    (1280, 1024),# SXGA
    (1600, 1200),# UXGA
    (1920, 1080),# Full HD
    (2560, 1440),# QHD
    (3840, 2160) # 4K UHD
]

print("Checking supported resolutions:")
for width, height in standard_resolutions:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if actual_width == width and actual_height == height:
        print(f"Supported: {width}x{height}")
    else:
        print(f"Not supported: {width}x{height}")

cap.release()
