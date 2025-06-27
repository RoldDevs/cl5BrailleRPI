# Raspberry Pi Braille Recognition and Display System

## Overview
This system uses a Raspberry Pi 4 Model B to capture images of text, convert them to braille patterns, and display them using servo motors. The system includes text-to-speech functionality and LCD display for user feedback.

## Hardware Components
- Raspberry Pi 4 Model B
- Logitech C930c Webcam
- 4x20 LCD Display (I2C interface)
- 16x SG90 Servo Motors
- Adafruit PCA9685 PWM Servo Driver
- LM2596S Speaker
- Tactile Push Button

## Installation Instructions

1. Install required system packages:
```
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install -y python3-pip python3-dev python3-opencv
sudo apt-get install -y libatlas-base-dev libhdf5-dev libhdf5-serial-dev
sudo apt-get install -y libjasper-dev libqtgui4 libqt4-test
sudo apt-get install -y espeak
```
2. Enable I2C interface on Raspberry Pi:
```
sudo raspi-config
```
3. Navigate to "Interface Options" > "I2C" > Enable
3. Install Python libraries:
```
python3.11 -m venv env
env/bin/activate
pip3 install -r requirements.txt
```
4. Connect hardware components according to the GPIO pin configuration in config.py:
- Connect the LCD display to the I2C pins (GPIO 2 & 3)
- Connect the PCA9685 to the I2C pins (GPIO 2 & 3)
- Connect the tactile button to GPIO 17
- Connect the speaker to GPIO 18
- Connect the webcam via USB
- Connect the servo motors to the PCA9685 board

5. Run the system:
```
(env) python main.py
``` 

## Hardware Setup Details

### LCD Display (4x20 I2C)
- VCC: 5V
- GND: GND
- SDA: GPIO 2
- SCL: GPIO 3

### PCA9685 Servo Driver
- VCC: 5V
- GND: GND
- SDA: GPIO 2
- SCL: GPIO 3
- V+: External 5V power supply (capable of powering all servos)

### Tactile Push Button
- One pin to GPIO 17
- Other pin to GND

### LM2596S Speaker
- Connected to GPIO 18 (PWM pin)
- GND to GND

### Servo Motors (SG90)
- Connect to the PCA9685 outputs
- Arrange in pairs to represent braille cells

## System Operation
1. Press the tactile button to capture an image
2. The system will process the image and extract text
3. The recognized text will be displayed on the LCD and spoken through the speaker
4. The braille pattern will then be displayed on the LCD
5. The text will be spoken again
6. The servo motors will move to represent the braille patterns

## Troubleshooting
- If the LCD display shows no text, check the I2C address in config.py
- If servos don't move, ensure the PCA9685 is properly powered
- If the camera doesn't work, check the camera index in config.py
- For text recognition issues, ensure good lighting conditions

## Required Python Libraries
The following libraries are required and will be installed via requirements.txt:
- opencv-python
- numpy
- tensorflow
- easyocr
- RPi.GPIO
- adafruit-blinka
- adafruit-circuitpython-pca9685
- adafruit-circuitpython-charlcd
- pyttsx3
- pillow

## Configuration
You can modify the system behavior by editing the config.py file:
- Adjust GPIO pin assignments
- Change LCD display settings
- Modify servo parameters
- Adjust text-to-speech settings

## License
This project is open source and available under the MIT License.
