import easyocr
import os
import cv2

def extract_text_from_image(image_path):
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found.")
        return

    # Load image using OpenCV (for better compatibility)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image '{image_path}'.")
        return

    # Initialize EasyOCR reader (only for English)
    reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if you have CUDA support

    # Perform OCR
    results = reader.readtext(image)

    # Extract only the text parts
    extracted_text = '\n'.join([text for _, text, _ in results])

    # Create output file path
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = f"{base_name}_output.txt"

    # Write to .txt file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(extracted_text)

    print(f"✅ Text extracted and saved to: {output_path}")

# Example usage:
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python easyocr_to_txt.py <image_path>")
    else:
        extract_text_from_image(sys.argv[1])
