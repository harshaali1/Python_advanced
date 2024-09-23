import cv2
import pytesseract
from googletrans import Translator

def handwritten_notes_translator(image_path, target_language="fr"):
    """
    Translates handwritten notes from an image into a target language.

    Args:
        image_path (str): The path to the input image file.
        target_language (str, optional): The target language code 
                                         (e.g., "fr" for French, 
                                          "es" for Spanish). 
                                          Defaults to "fr".

    Returns:
        str: The translated text or an error message if processing fails.
    """

    try:
        # 1. Image Preprocessing
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding for better edge detection
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 4)

        # Noise reduction (optional but recommended)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        # 2. Text Extraction using OCR (Tesseract)
        extracted_text = pytesseract.image_to_string(opening)

        # 3. Text Translation 
        translator = Translator()
        translation = translator.translate(extracted_text, dest=target_language)
        translated_text = translation.text

        return translated_text

    except Exception as e:
        return f"Error processing image: {str(e)}"


# Example usage:
image_file = "handwritten_notes.jpg"
translated_notes = handwritten_notes_translator(image_file, target_language="es") 

if "Error" not in translated_notes: 
    print(translated_notes)
else:
    print(translated_notes) 
