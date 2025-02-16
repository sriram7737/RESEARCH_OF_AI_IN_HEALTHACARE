import re
import logging
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
import pydicom
from PIL import Image
import pytesseract
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Initialize QA pipeline
try:
    tokenizer = AutoTokenizer.from_pretrained("./biobert_finetuned")
    model = AutoModelForQuestionAnswering.from_pretrained("./biobert_finetuned")
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
    logging.info("BioBERT QA pipeline initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing BioBERT QA pipeline: {e}")
    qa_pipeline = None

def answer_medical_question(question, prescription_context):
    """
    Answers a medical question using the prescription context if available.
    Falls back to default general medical context if no prescription context is provided.
    """
    default_context = (
        "This is general medical information. Conditions like cancer can present with symptoms "
        "such as unexplained weight loss, persistent fatigue, pain, or lumps. Always consult a "
        "healthcare professional for personalized advice."
    )
    chosen_context = prescription_context if prescription_context else default_context

    try:
        result = qa_pipeline(question=question, context=chosen_context)
        if isinstance(result, dict) and 'answer' in result:
            return {"answer": result['answer']}
        else:
            logging.error(f"Unexpected QA Pipeline result format: {result}")
            return {"error": "Unexpected QA Pipeline output structure."}
    except Exception as e:
        logging.error(f"Error answering medical question: {e}")
        return {"error": str(e)}

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'dcm'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_prescription(document):
    filename = document.filename.lower()
    logging.debug(f"Reading prescription: {filename}")
    try:
        if not allowed_file(filename):
            logging.error("File type not allowed.")
            return {"error": "File type not allowed"}

        if filename.endswith('.pdf'):
            import pdfplumber
            text = ""
            with pdfplumber.open(document) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return {"type": "text", "content": text}
        elif filename.endswith('.docx'):
            import docx
            doc = docx.Document(document)
            text = "\n".join([para.text for para in doc.paragraphs])
            return {"type": "text", "content": text}
        elif filename.endswith('.txt'):
            text = document.read().decode('utf-8', errors='ignore')
            return {"type": "text", "content": text}
        elif filename.endswith('.dcm'):
            return read_dicom_image(document)
        else:
            logging.error("Unsupported prescription file type.")
            return {"error": "Unsupported prescription file type."}
    except Exception as e:
        logging.error(f"Error reading prescription: {e}")
        return {"error": str(e)}

def read_dicom_image(file):
    try:
        dicom = pydicom.dcmread(file)
        pixel_array = dicom.pixel_array
        image = Image.fromarray(pixel_array)
        image.thumbnail((500, 500))
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        logging.debug("DICOM image processed and converted to base64")
        return {"type": "dicom", "content": image_base64}
    except Exception as e:
        logging.error(f"Error reading DICOM image: {e}")
        return {"error": str(e)}

def preprocess_text(text):
    # Clean up the text to remove unwanted patterns
    text = re.sub(r"Printed by \[.*?\] at .*?\n", "", text)
    text = re.sub(r"Page \d+ of \d+", "", text)
    text = text.replace(' - null', '\n')
    text = re.sub(r'[ \t]+', ' ', text)
    return text

def extract_field(text, pattern, flags=re.IGNORECASE | re.MULTILINE):
    match = re.search(pattern, text, flags)
    if match:
        group = match.group(1)
        if group:
            return group.strip()
    return "Not Found"

def extract_patient_info(text):
    name = extract_field(text, r"^Name[:\s]*(.*)$")
    date_of_birth = extract_field(text, r"^Date of Birth[:\s]*(.*)$")

    if name == "Not Found" or date_of_birth == "Not Found":
        match = re.search(r"^([\w\s]+) \(MRN: \d+\) \((\d{1,2}/\d{1,2}/\d{4})\)", text, re.MULTILINE)
        if match:
            name = match.group(1).strip()
            date_of_birth = match.group(2).strip()

    date_of_visit = extract_field(text, r"^Date of Visit[:\s]*(.*)$")
    doctor = extract_field(text, r"^Started by[:\s]*(.*)$")
    if doctor == "Not Found":
        doctor = extract_field(text, r"^Changed by[:\s]*(.*)$")
    visit_location = extract_field(text, r"^Visit Location[:\s]*(.*)$")
    reason_for_visit = extract_field(text, r"^For diagnoses[:\s]*(.*)$")
    return {
        "Name": name,
        "Date of Birth": date_of_birth,
        "Date of Visit": date_of_visit,
        "Doctor": doctor,
        "Visit Location": visit_location,
        "Reason for Visit": reason_for_visit
    }

def extract_medications(text):
    medications = []
    pattern = r"(?s)^Your Medication List.*?(?=^Education & Instructions|^Attached Information|^\s*$)"
    match = re.search(pattern, text, re.MULTILINE)
    if match:
        meds_text = match.group(0).strip()
        meds_lines = meds_text.split('\n')
        start_index = 0
        for i, line in enumerate(meds_lines):
            if 'As Needed Any Time' in line:
                start_index = i + 1
                break
        meds_lines = meds_lines[start_index:]
        current_med = {}
        for line in meds_lines:
            line = line.strip()
            if not line or line.startswith('*'):
                if current_med:
                    medications.append(current_med)
                    current_med = {}
                continue
            if 'Commonly known as:' in line:
                parts = line.split('Commonly known as:')
                name = parts[0].strip()
                common_name = parts[1].strip() if len(parts) > 1 else ''
                current_med['Name'] = name
                current_med['Common Name'] = common_name
            elif line.startswith('Take '):
                dosage = line.strip()
                current_med['Dosage'] = dosage
            elif 'For diagnoses:' in line:
                diagnosis = line.replace('For diagnoses:', '').strip()
                current_med['Diagnosis'] = diagnosis
            elif 'Started by:' in line or 'Changed by:' in line:
                doctor = line.replace('Started by:', '').replace('Changed by:', '').strip()
                current_med['Prescribed By'] = doctor
        if current_med:
            medications.append(current_med)
    else:
        logging.debug("Medications section not found.")
    return medications

def analyze_prescription(content):
    try:
        if content.get("type") == "dicom":
            return {"analysis": {"dicom_image": content.get("content")}}

        text = content.get("content", "")
        text = preprocess_text(text)
        logging.debug(f"Preprocessed Text:\n{text}")
        patient_info = extract_patient_info(text)
        medications = extract_medications(text)

        analysis = {
            "patient_info": patient_info,
            "medications": medications,
        }
        return {"analysis": analysis}
    except Exception as e:
        logging.error(f"Error analyzing prescription: {e}")
        return {"error": str(e)}
