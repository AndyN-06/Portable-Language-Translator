import board
import digitalio
import adafruit_ssd1306
from PIL import Image, ImageDraw, ImageFont
import time

class OLED_Display:
    def __init__(self, width=128, height=64):
        self.width = width
        self.height = height

        # Set up the SPI interface and display
        spi = board.SPI()
        self.oled_reset = digitalio.DigitalInOut(board.D25)  # Reset pin
        self.oled_dc = digitalio.DigitalInOut(board.D24)     # DC pin
        self.oled_cs = digitalio.DigitalInOut(board.D8)      # CS pin

        # Create the SSD1306 display class
        self.oled = adafruit_ssd1306.SSD1306_SPI(self.width, self.height,
spi, self.oled_dc, self.oled_reset, self.oled_cs)

        # Clear the display initially
        self.oled.fill(0)
        self.oled.show()

        self.image = Image.new("1", (self.width, self.height))
        self.draw = ImageDraw.Draw(self.image)

        self.font = ImageFont.load_default()

    def write_text(self, text, x=0, y=0):
        # Clear previous text
        self.draw.rectangle((0, 0, self.width, self.height), outline=0, fill=0)

        # Draw the new text
        self.draw.text((x, y), text, font=self.font, fill=255)

        # Display the image
        self.oled.image(self.image)
        self.oled.show()

    def clear_display(self):
        self.oled.fill(0)
        self.oled.show()

oled_display = OLED_Display()
oled_display.write_text("Hello, SSD1309!", 10, 20)
time.sleep(5)
oled_display.clear_display()