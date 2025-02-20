import time
import board
import busio
from adafruit_ssd1306 import SSD1306_SPI

# Set up SPI interface
spi = busio.SPI(clock=board.SCK, MOSI=board.MOSI)
cs = board.D5
dc = board.D6
reset = board.D4

display = SSD1306_SPI(128, 64, spi, dc, reset, cs)

display.fill(0)  # Clear display
display.text('Hello, Pi 5!', 0, 0)
display.show()

time.sleep(5)

