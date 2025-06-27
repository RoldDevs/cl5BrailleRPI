# Configuration file for Raspberry Pi GPIO pins and hardware settings

# GPIO Pin Configuration
PINS = {
    # I2C pins for PCA9685 and LCD Display
    "SCL": 3,  # GPIO 3 (SCL)
    "SDA": 2,  # GPIO 2 (SDA)
    
    # Push Button pin
    "BUTTON": 17,  # GPIO 17 for tactile push button
    
    # Speaker pin (PWM)
    "SPEAKER": 18,  # GPIO 18 for LM2596S Speaker
    
    # Camera settings
    "CAMERA_INDEX": 0  # Camera index (usually 0 for the first connected camera)
}

# LCD Display Configuration
LCD_CONFIG = {
    "ROWS": 4,
    "COLS": 20,
    "I2C_ADDR": 0x27  # Default I2C address for most 4x20 LCD displays (may need adjustment)
}

# PCA9685 Servo Controller Configuration
SERVO_CONFIG = {
    "FREQUENCY": 60,  # Hz
    "MIN_PULSE": 120,  # Min pulse length out of 4096
    "MAX_PULSE": 610,  # Max pulse length out of 4096
    "SERVO_COUNT": 16  # Number of servos connected
}

# Braille Display Configuration
BRAILLE_CONFIG = {
    "CHARS_PER_DISPLAY": 8,  # Maximum characters to display at once on the servo array
    "DISPLAY_TIME": 5  # Seconds to display each set of characters
}

# Audio Configuration
AUDIO_CONFIG = {
    "RATE": 150,  # Words per minute for text-to-speech
    "VOLUME": 0.8,  # Volume level (0.0 to 1.0)
    "LANGUAGE": "en"  # Language code for text-to-speech
}