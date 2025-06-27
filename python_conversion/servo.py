import time
import board
import busio
from adafruit_pca9685 import PCA9685

# Constants
SERVOMIN = 120   # Min pulse length out of 4096
SERVOMAX = 610   # Max pulse length out of 4096
SERVO_FREQ = 60  # Hz

# Setup I2C and PCA9685
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = SERVO_FREQ

def angle_to_pulse(angle):
    """Map angle (0-180) to PWM pulse."""
    return int((angle / 180.0) * (SERVOMAX - SERVOMIN) + SERVOMIN)

def reset_servos():
    """Set all servos to neutral position (180°)."""
    for i in range(16):
        pca.channels[i].duty_cycle = angle_to_pulse(180)
        time.sleep(0.1)

def execute_servo_commands(letter, start_index=0):
    """Set servo positions based on the letter.
    
    Args:
        letter: The letter to represent in braille
        start_index: The starting index for the servo channels
    """
    mapping = {
        'a': (5, 180), 'b': (35, 180), 'c': (5, 165), 'd': (5, 75),
        'e': (5, 57), 'f': (35, 165), 'g': (35, 75), 'h': (35, 57),
        'i': (57, 165), 'j': (57, 75), 'k': (140, 180), 'l': (110, 180),
        'm': (140, 165), 'n': (140, 75), 'o': (140, 57), 'p': (110, 165),
        'q': (110, 75), 'r': (110, 57), 's': (75, 165), 't': (75, 75),
        'u': (140, 5), 'v': (110, 5), 'w': (57, 110), 'x': (140, 140),
        'y': (140, 110), 'z': (140, 35)
    }

    angle1, angle2 = mapping.get(letter.lower(), (180, 180))
    pca.channels[start_index].duty_cycle = angle_to_pulse(angle1)
    time.sleep(0.1)
    pca.channels[start_index + 1].duty_cycle = angle_to_pulse(angle2)
    time.sleep(0.1)

def process_input(text, start_index=0):
    """Process a string of text and set servo positions.
    
    Args:
        text: The text to convert to braille
        start_index: The starting index for the servo channels
    """
    current_index = start_index
    for char in text:
        if char.isalpha():
            execute_servo_commands(char, current_index)
            current_index += 2
            if current_index >= 16:
                # We've used all available servos
                break

if __name__ == "__main__":
    try:
        print("Braille Servo System Ready. Enter letters:")
        while True:
            user_input = input(">> ")
            reset_servos()
            process_input(user_input.strip())
    except KeyboardInterrupt:
        print("Exiting...")
        reset_servos()
        pca.deinit()
