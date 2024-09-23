import pytesseract
import cv2
import pandas as pd

# Configure Tesseract path if not in system PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Replace with your actual path

def ocr_and_format(image_path):
    """
    Performs OCR on a scanned document and attempts to extract structured data.

    Args:
        image_path (str): Path to the scanned document image.

    Returns:
        str: Extracted text from the document.
        list: List of DataFrames if tables are detected, otherwise an empty list.
    """
    # Read the image
    img = cv2.imread(image_path)
    
    if img is None:
        raise FileNotFoundError(f"Image at {image_path} not found.")

    # Preprocess the image for better OCR (optional but recommended)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Perform OCR using Tesseract
    text = pytesseract.image_to_string(thresh)

    # Table extraction (experimental, may require refinement)
    tables = []
    try:
        df = pd.read_fwf(pd.compat.StringIO(text))
        if len(df.columns) > 1:  # Basic check for table structure
            tables.append(df)
    except Exception as e:
        print(f"Table extraction failed: {e}")

    return text, tables

# Example usage
image_path = 'img.png'  # Replace with your image path
text, tables = ocr_and_format(image_path)

# Print extracted text
print("Extracted Text:\n", text)

# Print tables if found
if tables:
  for i, table in enumerate(tables):
      print(f"\nTable {i+1}:")
      print(table)
else:
  print("\nNo tables detected.")