import os
import cv2
import numpy as np
import easyocr
import time
import tensorflow as tf
from braille_utils import magic_filter_bw, text_to_braille, create_visualization
from cnn_validator import BrailleValidator

# Create necessary directories if they don't exist
output_dir = 'output'
first_val_dir = 'first_val'
second_val_dir = 'second_val'
output_braille_dir = 'output_braille'

for directory in [output_dir, first_val_dir, second_val_dir, output_braille_dir]:
    os.makedirs(directory, exist_ok=True)

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False, model_storage_directory='./easyocr_models')

# Initialize the CNN validator
validator = BrailleValidator(model_path='braille_cnn_model.h5' if os.path.exists('braille_cnn_model.h5') else None)

# === UPDATED IMAGE ENHANCEMENT AND UTILITY FUNCTIONS ===

def enhance_image_for_ocr(image):
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
    return magic_filter_bw(image)

def save_text_to_file(text, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)

def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def capture_and_recognize():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    print("Press 'c' to capture an image or 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        cv2.imshow('Camera Feed', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c'):
            process_image(frame)
    cap.release()
    cv2.destroyAllWindows()

def process_image(image):
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
    results = reader.readtext(enhanced_img, paragraph=True)

    recognized_text = " ".join([text for _, text in results]).strip()


    first_val_path = os.path.join(first_val_dir, "easyocr_textrecognition.txt")
    save_text_to_file(recognized_text, first_val_path)

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

    braille_text = text_to_braille(verified_text)
    output_path = os.path.join(output_dir, f"result_{timestamp}.png")
    visualization = create_visualization(verified_text, braille_text, output_path)

    braille_output_path = os.path.join(output_braille_dir, f"{timestamp}.txt")
    with open(braille_output_path, 'w', encoding='utf-8') as f:
        f.write(f"Original Text: {recognized_text}\n\n")
        f.write(f"Recognized Text: {verified_text}\n\n")
        f.write(f"Braille Pattern: {braille_text}\n")

    print("\nResults:")
    print(f"Original Image: {original_path}")
    print(f"Processed Image: {processed_path}")
    print(f"First Validation (EasyOCR): {first_val_path}")
    print(f"Second Validation (ML): {second_val_path}")
    print(f"Braille Output: {braille_output_path}")
    print(f"Result Image: {output_path}")

    cv2.imshow('Result', np.array(visualization))
    cv2.waitKey(0)

def main():
    print("Dual Validation Text to Braille Conversion System")
    print("==============================================")
    while True:
        print("\nOptions:")
        print("1. Capture from camera")
        print("2. Load image from file")
        print("3. Exit")
        choice = input("Enter your choice (1-3): ")
        if choice == '1':
            capture_and_recognize()
        elif choice == '2':
            image_path = input("Enter the path to the image file: ")
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                process_image(image)
            else:
                print("Error: File not found.")
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
