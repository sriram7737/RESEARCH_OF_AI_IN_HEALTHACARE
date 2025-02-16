import pytesseract
from PIL import Image
import io

def analyze_image(image):
    image = Image.open(io.BytesIO(image.read()))
    text = pytesseract.image_to_string(image)
    return {"text": text}
