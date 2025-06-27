import os
import cv2
import numpy as np
import time
import RPi.GPIO as GPIO
import board
import busio
import adafruit_character_lcd.character_lcd_rgb_i2c as character_lcd
from adafruit_pca9685 import PCA9685
import pyttsx3
from PIL import Image

# Import project modules
from braille_utils import magic_filter_bw, text_to_braille, text_to_binary_braille
from cnn_validator import BrailleValidator
import config

# Create necessary directories if they don't exist
output_dir = 'output'
first_val_dir = 'first_val'
second_val_dir = 'second_val'
output_braille_dir = 'output_braille'

for directory in [output_dir, first_val_dir, second_val_dir, output_braille_dir]:
    os.makedirs(directory, exist_ok=True)

# Initialize GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(config.PINS["BUTTON"], GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Initialize I2C for LCD and PCA9685
i2c = busio.I2C(board.SCL, board.SDA)

# Initialize LCD Display
lcd = character_lcd.Character_LCD_RGB_I2C(
    i2c, 
    config.LCD_CONFIG["COLS"], 
    config.LCD_CONFIG["ROWS"], 
    address=config.LCD_CONFIG["I2C_ADDR"]
)
lcd.clear()
lcd.color = [100, 100, 100]  # Set backlight color

# Initialize PCA9685 for servo control
pca = PCA9685(i2c)
pca.frequency = config.SERVO_CONFIG["FREQUENCY"]

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', config.AUDIO_CONFIG["RATE"])
engine.setProperty('volume', config.AUDIO_CONFIG["VOLUME"])

# Initialize the OCR reader
try:
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False, model_storage_directory='./easyocr_models')
except ImportError:
    print("EasyOCR not installed. Please install it using: pip install easyocr")
    exit(1)

# Initialize the CNN validator
validator = BrailleValidator(model_path='braille_cnn_model.h5' if os.path.exists('braille_cnn_model.h5') else None)

# === Utility Functions ===

def angle_to_pulse(angle):
    """Map angle (0-180) to PWM pulse."""
    return int((angle / 180.0) * (config.SERVO_CONFIG["MAX_PULSE"] - config.SERVO_CONFIG["MIN_PULSE"]) + config.SERVO_CONFIG["MIN_PULSE"])

def reset_servos():
    """Set all servos to neutral position (180°)."""
    for i in range(config.SERVO_CONFIG["SERVO_COUNT"]):
        pca.channels[i].duty_cycle = angle_to_pulse(180)
        time.sleep(0.05)

def display_braille_pattern(binary_pattern):
    """Display a binary braille pattern on the servo array.
    
    Args:
        binary_pattern: List of 6-element lists representing braille dots
    """
    reset_servos()
    
    # We can display up to 8 characters (using 16 servos, 2 per character)
    display_chars = min(len(binary_pattern), config.BRAILLE_CONFIG["CHARS_PER_DISPLAY"])
    
    for i in range(display_chars):
        char_pattern = binary_pattern[i]
        
        # Each character uses 2 servos
        servo_index = i * 2
        
        if servo_index < config.SERVO_CONFIG["SERVO_COUNT"]:
            # First servo (dots 1-3)
            angle1 = 5 if char_pattern[0] else 180  # Dot 1
            angle2 = 5 if char_pattern[1] else 180  # Dot 2
            angle3 = 5 if char_pattern[2] else 180  # Dot 3
            
            # Set position for first servo based on dots 1-3
            # This is a simplified approach - you may need to adjust
            # based on your specific servo arrangement
            if char_pattern[0] and char_pattern[1] and char_pattern[2]:
                pca.channels[servo_index].duty_cycle = angle_to_pulse(5)  # All dots raised
            elif not char_pattern[0] and not char_pattern[1] and not char_pattern[2]:
                pca.channels[servo_index].duty_cycle = angle_to_pulse(180)  # No dots raised
            else:
                # Calculate a position based on which dots are raised
                avg_angle = 180 - (((angle1 == 5) + (angle2 == 5) + (angle3 == 5)) * 58)
                pca.channels[servo_index].duty_cycle = angle_to_pulse(avg_angle)
        
        if servo_index + 1 < config.SERVO_CONFIG["SERVO_COUNT"]:
            # Second servo (dots 4-6)
            angle4 = 5 if char_pattern[3] else 180  # Dot 4
            angle5 = 5 if char_pattern[4] else 180  # Dot 5
            angle6 = 5 if char_pattern[5] else 180  # Dot 6
            
            # Set position for second servo based on dots 4-6
            if char_pattern[3] and char_pattern[4] and char_pattern[5]:
                pca.channels[servo_index + 1].duty_cycle = angle_to_pulse(5)  # All dots raised
            elif not char_pattern[3] and not char_pattern[4] and not char_pattern[5]:
                pca.channels[servo_index + 1].duty_cycle = angle_to_pulse(180)  # No dots raised
            else:
                # Calculate a position based on which dots are raised
                avg_angle = 180 - (((angle4 == 5) + (angle5 == 5) + (angle6 == 5)) * 58)
                pca.channels[servo_index + 1].duty_cycle = angle_to_pulse(avg_angle)
        
        time.sleep(0.1)

def display_text_on_lcd(text):
    """Display text on the LCD screen."""
    lcd.clear()
    
    # Limit text to fit on the display
    max_chars = config.LCD_CONFIG["COLS"]
    max_rows = config.LCD_CONFIG["ROWS"]
    
    # Split text into lines
    words = text.split()
    lines = []
    current_line = ""
    
    for word in words:
        if len(current_line) + len(word) + 1 <= max_chars:
            if current_line:
                current_line += " " + word
            else:
                current_line = word
        else:
            lines.append(current_line)
            current_line = word
    
    if current_line:
        lines.append(current_line)
    
    # Display lines on LCD
    for i, line in enumerate(lines[:max_rows]):
        lcd.cursor_position(0, i)
        lcd.message = line

def display_braille_on_lcd(braille_text):
    """Display braille pattern on the LCD screen."""
    lcd.clear()
    
    # Limit text to fit on the display
    max_chars = config.LCD_CONFIG["COLS"]
    max_rows = config.LCD_CONFIG["ROWS"]
    
    # Split text into lines of max_chars length
    lines = [braille_text[i:i+max_chars] for i in range(0, len(braille_text), max_chars)]
    
    # Display lines on LCD
    for i, line in enumerate(lines[:max_rows]):
        lcd.cursor_position(0, i)
        lcd.message = line

def speak_text(text):
    """Convert text to speech and play it."""
    engine.say(text)
    engine.runAndWait()

def enhance_image_for_ocr(image):
    """Enhance image for better OCR results."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    dilated = cv2.dilate(opening, kernel, iterations=1)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced, dilated

def preprocess_image(image):
    """Preprocess image for better text recognition."""
    return magic_filter_bw(image)

def save_text_to_file(text, file_path):
    """Save text to a file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)

def read_text_from_file(file_path):
    """Read text from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def process_image(image):
    """Process an image to extract text and convert to braille."""
    # Display status on LCD
    lcd.clear()
    lcd.message = "Processing image..."
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    processed_img = preprocess_image(image)
    original_path = os.path.join(output_dir, f"original_{timestamp}.jpg")
    processed_path = os.path.join(output_dir, f"processed_{timestamp}.jpg")
    cv2.imwrite(original_path, image)
    cv2.imwrite(processed_path, processed_img)
    enhanced_img, binary_img = enhance_image_for_ocr(image)
    enhanced_path = os.path.join(output_dir, f"enhanced_{timestamp}.jpg")
    binary_path = os.path.join(output_dir, f"binary_{timestamp}.jpg")
    cv2.imwrite(enhanced_path, enhanced_img)
    cv2.imwrite(binary_path, binary_img)

    # OCR with paragraph mode for better sentence structure
    lcd.clear()
    lcd.message = "Recognizing text..."
    results = reader.readtext(enhanced_img, paragraph=True)

    recognized_text = " ".join([text for _, text in results]).strip()

    first_val_path = os.path.join(first_val_dir, "easyocr_textrecognition.txt")
    save_text_to_file(recognized_text, first_val_path)

    # Validate text with CNN
    lcd.clear()
    lcd.message = "Validating text..."
    verified_text = ""
    lines = recognized_text.split('\n')
    for line in lines:
        verified_line = ""
        words = line.split()
        for word in words:
            verified_word = ""
            for char in word.upper():
                if 'A' <= char <= 'Z':
                    char_img = np.zeros((64, 64, 1), dtype=np.uint8)
                    cv2.putText(char_img, char, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
                    predicted_char, confidence = validator.predict(char_img)
                    verified_word += predicted_char if confidence > 0.5 else char
                else:
                    verified_word += char
            verified_line += verified_word + " "
        verified_text += verified_line.strip() + "\n"

    verified_text = verified_text.strip()
    second_val_path = os.path.join(second_val_dir, "machine_textrecognition.txt")
    save_text_to_file(verified_text, second_val_path)

    # Convert to braille
    braille_text = text_to_braille(verified_text)
    braille_output_path = os.path.join(output_braille_dir, f"{timestamp}.txt")
    with open(braille_output_path, 'w', encoding='utf-8') as f:
        f.write(f"Original Text: {recognized_text}\n\n")
        f.write(f"Recognized Text: {verified_text}\n\n")
        f.write(f"Braille Pattern: {braille_text}\n")

    # Display recognized text on LCD
    lcd.clear()
    display_text_on_lcd(verified_text)
    
    # Speak the recognized text
    speak_text(verified_text)
    
    # Display braille pattern on LCD
    time.sleep(1)  # Short pause
    lcd.clear()
    display_braille_on_lcd(braille_text)
    
    # Speak the text again
    speak_text(verified_text)
    
    # Convert text to binary braille representation for servo control
    binary_braille = text_to_binary_braille(verified_text)
    
    # Display braille pattern on servo array
    # If text is longer than what can be displayed at once, show it in chunks
    chunks = [binary_braille[i:i+config.BRAILLE_CONFIG["CHARS_PER_DISPLAY"]] 
              for i in range(0, len(binary_braille), config.BRAILLE_CONFIG["CHARS_PER_DISPLAY"])]
    
    for chunk in chunks:
        display_braille_pattern(chunk)
        time.sleep(config.BRAILLE_CONFIG["DISPLAY_TIME"])
    
    # Reset servos when done
    reset_servos()
    
    # Return to ready state
    lcd.clear()
    lcd.message = "Ready for next scan"

def main():
    """Main function to run the braille recognition and display system."""
    # Initialize camera
    cap = cv2.VideoCapture(config.PINS["CAMERA_INDEX"])
    if not cap.isOpened():
        lcd.clear()
        lcd.message = "Error: Camera not\nfound!"
        time.sleep(3)
        return
    
    # Display welcome message
    lcd.clear()
    lcd.message = "Braille Recognition\nSystem Ready"
    lcd.color = [0, 100, 0]  # Green backlight
    
    # Reset servos to neutral position
    reset_servos()
    
    try:
        button_pressed = False
        while True:
            # Check if button is pressed
            button_state = GPIO.input(config.PINS["BUTTON"])
            
            if button_state == GPIO.LOW and not button_pressed:
                # Button pressed
                button_pressed = True
                lcd.color = [100, 0, 0]  # Red backlight during capture
                
                # Capture image
                ret, frame = cap.read()
                if ret:
                    # Process the captured image
                    process_image(frame)
                else:
                    lcd.clear()
                    lcd.message = "Error: Failed to\ncapture image!"
                    time.sleep(2)
                
                lcd.color = [0, 100, 0]  # Back to green when done
            
            elif button_state == GPIO.HIGH and button_pressed:
                # Button released
                button_pressed = False
            
            # Small delay to prevent CPU hogging
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\nExiting...")
    
    finally:
        # Clean up
        lcd.clear()
        lcd.color = [0, 0, 0]  # Turn off backlight
        cap.release()
        reset_servos()
        pca.deinit()
        GPIO.cleanup()

if __name__ == "__main__":
    main()