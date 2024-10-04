import json
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm

# Define page parameters
PAGE_PARAMS = {
    "width": 210 * mm,
    "height": 297 * mm
}

# Define a class to manage drawing positions
class PositionManager:
    def __init__(self, data_type, page_number):
        self.data_type = data_type
        self.page_number = page_number
        self.positions = self._load_positions()

    def _load_positions(self):
        # Load position data from JSON file based on data type and page number
        with open(f'position_data/_{self.data_type}_page_{self.page_number}.json', 'r') as f:
            return json.load(f)

    def get_position(self, key):
        # Retrieve position coordinates by key
        return self.positions.get(key, (0, 0))

# Function to draw text on the PDF canvas
def draw_text(canvas, text, x, y, font_size=12):
    canvas.setFont("Helvetica", font_size)
    canvas.drawString(x * mm, y * mm, text)

# Function to draw data on the PDF based on type and page number
def draw_data_on_page(canvas, data, data_type, page_number):
    position_manager = PositionManager(data_type, page_number)

    # Retrieve and draw data based on position keys
    for key, value in data.items():
        x, y = position_manager.get_position(key)
        draw_text(canvas, str(value), x, y)

# Generate the PDF document
def generate_pdf(data, output_filename):
    pdf = canvas.Canvas(output_filename, pagesize=(PAGE_PARAMS["width"], PAGE_PARAMS["height"]))

    # Page 1: vict
    pdf.showPage()
    draw_data_on_page(pdf, data.get("vict", {}), "vict", 1)

    # Page 2: commercial
    pdf.showPage()
    draw_data_on_page(pdf, data.get("commercial", {}), "commercial", 2)

    pdf.save()

if __name__ == "__main__":
    # Load data from JSON file
    with open(  "data.json", 'r') as f:
        data = json.load(f)

    # Generate the PDF
    generate_pdf(data, 'output.pdf')
