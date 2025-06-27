import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# Braille patterns dictionary (mapping from letters to braille patterns)
braille_patterns = {
    'A': '⠁', 'B': '⠃', 'C': '⠉', 'D': '⠙', 'E': '⠑',
    'F': '⠋', 'G': '⠛', 'H': '⠓', 'I': '⠊', 'J': '⠚',
    'K': '⠅', 'L': '⠇', 'M': '⠍', 'N': '⠝', 'O': '⠕',
    'P': '⠏', 'Q': '⠟', 'R': '⠗', 'S': '⠎', 'T': '⠞',
    'U': '⠥', 'V': '⠧', 'W': '⠺', 'X': '⠭', 'Y': '⠽',
    'Z': '⠵', ' ': '⠀'
}

# Binary representation of braille patterns (for servo motor control)
# Each braille cell has 6 dots arranged in a 2x3 grid
# 1 means dot is raised, 0 means dot is not raised
braille_binary = {
    'A': [1, 0, 0, 0, 0, 0], 'B': [1, 1, 0, 0, 0, 0],
    'C': [1, 0, 0, 1, 0, 0], 'D': [1, 0, 0, 1, 1, 0],
    'E': [1, 0, 0, 0, 1, 0], 'F': [1, 1, 0, 1, 0, 0],
    'G': [1, 1, 0, 1, 1, 0], 'H': [1, 1, 0, 0, 1, 0],
    'I': [0, 1, 0, 1, 0, 0], 'J': [0, 1, 0, 1, 1, 0],
    'K': [1, 0, 1, 0, 0, 0], 'L': [1, 1, 1, 0, 0, 0],
    'M': [1, 0, 1, 1, 0, 0], 'N': [1, 0, 1, 1, 1, 0],
    'O': [1, 0, 1, 0, 1, 0], 'P': [1, 1, 1, 1, 0, 0],
    'Q': [1, 1, 1, 1, 1, 0], 'R': [1, 1, 1, 0, 1, 0],
    'S': [0, 1, 1, 1, 0, 0], 'T': [0, 1, 1, 1, 1, 0],
    'U': [1, 0, 1, 0, 0, 1], 'V': [1, 1, 1, 0, 0, 1],
    'W': [0, 1, 0, 1, 1, 1], 'X': [1, 0, 1, 1, 0, 1],
    'Y': [1, 0, 1, 1, 1, 1], 'Z': [1, 0, 1, 0, 1, 1],
    ' ': [0, 0, 0, 0, 0, 0]
}

def text_to_braille(text):
    """Convert text to braille characters"""
    text = text.upper()
    braille_text = ''
    for char in text:
        if char in braille_patterns:
            braille_text += braille_patterns[char]
        else:
            braille_text += char  # Keep non-mapped characters as is
    return braille_text

def text_to_binary_braille(text):
    """Convert text to binary braille representation for servo control"""
    text = text.upper()
    binary_representation = []
    for char in text:
        if char in braille_binary:
            binary_representation.append(braille_binary[char])
        else:
            binary_representation.append(braille_binary[' '])  # Use space for unknown chars
    return binary_representation

# === Enhanced Image Preprocessing Functions ===
def resize_to_64x64(image):
    """Resize character to 64x64 while preserving aspect ratio"""
    h, w = image.shape
    if h == 0 or w == 0:
        return np.zeros((64, 64), dtype=np.uint8)
    scale = 20.0 / max(w, h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((64, 64), dtype=np.uint8)
    x_offset = (64 - new_w) // 2
    y_offset = (64 - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    _, canvas = cv2.threshold(canvas, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return canvas

def magic_filter_bw(image):
    """CamScanner-style preprocessing for better text extraction"""
    # Convert to grayscale if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # Apply bilateral filter to reduce noise while preserving edges
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 15, 15)
    
    # Apply morphological operations to clean up the image
    kernel = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Sharpen the image
    sharpened = cv2.GaussianBlur(opened, (0, 0), 3)
    result = cv2.addWeighted(opened, 1.5, sharpened, -0.5, 0)
    
    return result

def enhanced_segment_characters(image):
    """Enhanced character segmentation with improved preprocessing"""
    # Apply the magic filter to clean up the image
    clean_bw = magic_filter_bw(image)
    
    # Apply binary thresholding
    binary = cv2.threshold(clean_bw, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Find connected components
    num_labels, labels_im = cv2.connectedComponents(binary, connectivity=8)
    
    # Extract character regions
    char_images = []
    boxes = []
    
    for label in range(1, num_labels):
        mask = (labels_im == label).astype(np.uint8) * 255
        x, y, w, h = cv2.boundingRect(mask)
        
        # Filter out very small contours
        if w >= 10 and h >= 10:
            # Extract the character
            char_img = binary[y:y+h, x:x+w]
            
            # Resize to 28x28 while preserving aspect ratio
            processed_char = resize_to_64x64(char_img)
            
            # Normalize for model input
            normalized = processed_char.astype("float32") / 255.0
            
            char_images.append((normalized, (x, y, w, h)))
            boxes.append((x, y, w, h))
    
    # Sort characters from left to right
    if char_images:
        # Sort by x-coordinate
        char_images = [x for _, x in sorted(zip(boxes, char_images), key=lambda pair: pair[0][0])]
    
    return char_images

# Keep the original functions for backward compatibility
def preprocess_image(image, target_size=(64, 64)):
    """Preprocess image for model prediction (original method)"""
    # Convert to grayscale if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # Apply thresholding to handle different lighting conditions
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours to isolate characters
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours found, return the original image resized
    if not contours:
        return cv2.resize(gray, target_size)
    
    # Find bounding box around all contours
    x_min, y_min, x_max, y_max = float('inf'), float('inf'), 0, 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
    
    # Add padding
    padding = 5
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(gray.shape[1], x_max + padding)
    y_max = min(gray.shape[0], y_max + padding)
    
    # Crop the image to the bounding box
    cropped = gray[y_min:y_max, x_min:x_max]
    
    # Resize to target size
    resized = cv2.resize(cropped, target_size)
    
    # Normalize
    normalized = resized / 255.0
    
    # Expand dimensions for model input
    return normalized

def segment_characters(image):
    """Segment characters from an image (original method)"""
    # Convert to grayscale if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Apply dilation to connect components
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours from left to right
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])
    
    # Extract character regions
    char_images = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter out very small contours
        if w > 5 and h > 5:
            char_img = gray[y:y+h, x:x+w]
            # Preprocess for model
            processed_char = preprocess_image(char_img)
            char_images.append((processed_char, (x, y, w, h)))
    
    return char_images

def create_visualization(original_text, braille_text, output_path='output.png'):
    """Create a visualization image with original text and braille pattern"""
    # Create a white image
    img_width = max(len(original_text), len(braille_text)) * 50 + 100
    img_height = 300
    img = Image.new('RGB', (img_width, img_height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to load a font that supports braille
    try:
        # For Windows, try a common font that supports braille
        font = ImageFont.truetype("segoeui.ttf", 40)
    except IOError:
        # Fallback to default
        font = ImageFont.load_default()
    
    # Draw original text
    draw.text((50, 50), f"Original: {original_text}", fill='black', font=font)
    
    # Draw braille text
    draw.text((50, 150), f"Braille: {braille_text}", fill='black', font=font)
    
    # Save the image
    img.save(output_path)
    return img