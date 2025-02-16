# app.py

import logging
import os
import re
import asyncio
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, LlamaForCausalLM
from peft import PeftModel
from dotenv import load_dotenv
import torch

# Local service imports (ensure these are correctly implemented or adjust as needed)
from services.image_service import analyze_image
from services.voice_service import speech_to_text, text_to_speech
from services.business_service import get_business_info
from services.document_service import read_document, analyze_document
from services.prescription_service import read_prescription, analyze_prescription
from services.scraping_service import scrape_website

app = Flask(__name__)

# ------------------------------------------------------------------------------
# SECURITY SETUP
# ------------------------------------------------------------------------------
# Load environment variables from .env file
load_dotenv()

# ------------------------------------------------------------------------------
# GLOBAL CONFIG / MODEL LOADING
# ------------------------------------------------------------------------------
MODEL_PATH = r"C:\Users\srira\OneDrive\Desktop\chatbot_collab\ree_fine_tuned_llama"  # Replace with your actual model path
FEEDBACK_FILE = "feedback.log"

# Determine device (GPU if available, else CPU)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize model and tokenizer outside of the try/except so it can be called later
tokenizer = None
model = None

try:
    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    base_model = LlamaForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True).to(device)

    # Verify and log eos_token_id
    eos_token_id = tokenizer.eos_token_id
    eos_token = tokenizer.decode([eos_token_id])
    logging.debug(f"Tokenizer eos_token_id: {eos_token_id}, Token: {repr(eos_token)}")

    # Load PEFT adapters (Ensure only one adapter is loaded)
    peft_model_path = MODEL_PATH  # Adjust if adapters are saved separately
    model = PeftModel.from_pretrained(
        base_model,
        peft_model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    model.to(device)
    model.eval()
    logging.info("PEFT adapters loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model or tokenizer: {e}")
    # Set model to None in case of error
    model = None



# ------------------------------------------------------------------------------
# GLOBAL VARIABLES / CONTEXT
# ------------------------------------------------------------------------------
context_data = ""
prescription_context = ""

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'dcm'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def requires_factual_answer(prompt: str) -> bool:
    """
    Determines if the prompt is a factual question based on punctuation.
    """
    return prompt.strip().endswith('?')

def duckduckgo_instant_answer(query):
    """
    Retrieve an instant answer from DuckDuckGo if available.
    """
    try:
        import requests
        params = {
            'q': query,
            'format': 'json',
            'no_redirect': 1,
            'no_html': 1,
            'skip_disambig': 1
        }
        response = requests.get('https://api.duckduckgo.com/', params=params, timeout=5)
        response.raise_for_status()
        data = response.json()

        if data.get('AbstractText'):
            return data['AbstractText']
        elif data.get('RelatedTopics'):
            related = data['RelatedTopics']
            if isinstance(related, list) and len(related) > 0 and 'Text' in related[0]:
                return related[0]['Text']
        return None
    except Exception as e:
        logging.error(f"Error fetching DuckDuckGo instant answer: {e}")
        return None

def custom_web_scrape(query):
    """
    Fallback custom web scraping via DuckDuckGo SERP snippet.
    """
    try:
        import requests
        from bs4 import BeautifulSoup
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        search_url = f"https://duckduckgo.com/html/?q={requests.utils.quote(query)}"
        response = requests.get(search_url, headers=headers, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Grab the first snippet
        snippet = ''
        result = soup.find('a', class_='result__a')
        if result:
            parent = result.find_parent('div', class_='result')
            if parent:
                snippet_div = parent.find('a', class_='result__snippet')
                if snippet_div:
                    snippet = snippet_div.get_text(separator=' ', strip=True)

        return snippet if snippet else None
    except Exception as e:
        logging.error(f"Error during custom web scraping: {e}")
        return None

def clean_response(response: str) -> str:
    """
    Cleans and truncates the generated text to extract the relevant answer.
    """
    # Find the position of 'Answer:' and extract text after it
    if "Answer:" in response:
        answer = response.split("Answer:")[-1].strip()
    else:
        answer = response.strip()

    # Remove any unintended repetitions or patterns
    # Example: Replace repeated words like 'cancer cancer cancer' with a single 'cancer'
    answer = re.sub(r'\b(\w+)\s+\1+\b', r'\1', answer, flags=re.IGNORECASE)

    # Further cleaning if necessary
    answer = re.sub(r'\s+', ' ', answer)  # Replace multiple spaces with single space
    
    # Remove any leading or trailing non-alphanumeric characters
    answer = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', answer)
    
    return answer


def post_process_answer(generated_text: str) -> str:
    """
    Cleans and truncates the generated text to extract the relevant answer.
    """
    # Split the generated text by "Answer:" and take the last part
    parts = generated_text.split("Answer:")
    if len(parts) > 1:
        answer = parts[-1].strip()
    else:
        answer = generated_text.strip()
    
    answer = clean_response(answer)
    return answer

def generate_response(prompt: str, max_new_tokens=100, min_new_tokens=30, temperature=0.6, top_p=0.9) -> str:
    """
    Generates a response from the fine-tuned LLaMA model based on the input prompt.

    Args:
        prompt (str): The input prompt/question.
        max_new_tokens (int): Maximum number of new tokens to generate.
        min_new_tokens (int): Minimum number of new tokens to generate.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling probability.

    Returns:
        str: The generated response.
    """
    if not model:
        return "Model is not available."

    try:
        logging.debug(f"Generating response for prompt: {repr(prompt)}")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        logging.debug(f"Tokenized inputs: {inputs}")

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            temperature=temperature,
            top_p=top_p,
            num_beams=3,
            repetition_penalty=1.5,
            no_repeat_ngram_size=4,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True
        )
        logging.debug(f"Generated token IDs: {outputs}")

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.debug(f"Raw generated text: {repr(generated_text)}")

        # Post-process the generated text
        answer = post_process_answer(generated_text)
        logging.debug(f"Final QA answer: {repr(answer)}")
        return answer
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return "I'm sorry, there was a problem generating an answer."


def handle_fact_question(question: str) -> str:
    """
    Retrieve context (Wikipedia → DuckDuckGo → custom scrape),
    then produce a QA-style answer with the text-generation pipeline.
    """
    try:
        import wikipedia
        # 1. Attempt Wikipedia lookup
        try:
            search_query = question.strip().rstrip("?")
            search_query = re.sub(r'[^a-zA-Z0-9\s]', '', search_query)
            page = wikipedia.page(search_query)
            context = page.summary
        except Exception as e:
            logging.info(f"Wikipedia lookup failed: {e}")

            # 2. DuckDuckGo fallback
            search_results = duckduckgo_instant_answer(question)
            if not search_results:
                # 3. Last resort: custom web scrape
                search_results = custom_web_scrape(question)
                if not search_results:
                    return "I'm sorry, I couldn't find any information on that."
                context = search_results
            else:
                context = search_results

        # Log the retrieved context
        logging.debug(f"Context retrieved: {repr(context)}")

        # 4. Generate an answer from the gathered context
        prompt = f"""
You are a helpful medical assistant with expertise in health and medicine.

Below is some contextual information:

{context}

The user asked: {question}

Provide a concise, factual response based on the context above.
Answer:
""".strip()

        answer = generate_response(prompt)

        # Return a cleaned-up version
        return post_process_answer(answer)

    except Exception as e:
        logging.error(f"Error in handle_fact_question: {e}")
        return "An error occurred while fetching information."

# ------------------------------------------------------------------------------
# MEDICAL QUESTION ANSWERING USING FINE-TUNED LLAMA MODEL
# ------------------------------------------------------------------------------
def answer_medical_question(question: str, prescription_context: str) -> dict:
    """
    Answers a medical question using the prescription context if available.
    Utilizes the fine-tuned LLaMA model to generate the answer.

    Args:
        question (str): The medical question to answer.
        prescription_context (str): Context from the prescription.

    Returns:
        dict: A dictionary containing the answer or error information.
    """
    default_context = (
        "This is general medical information. Conditions like cancer can present with symptoms "
        "such as unexplained weight loss, persistent fatigue, pain, or lumps. Always consult a "
        "healthcare professional for personalized advice."
    )
    chosen_context = prescription_context if prescription_context else default_context

    try:
        prompt = (
            f"""You are a helpful medical assistant and a top general doctor. If any health issues arise, you are the first person to consult for help. If you are asked questions, you need to provide concise and accurate answers.
            
Please answer the question below in 1-2 sentences, accurately and concisely.
            
Question: {question}
Answer:
            """.strip()
        )

        answer = generate_response(prompt)

        return {"answer": answer}
    except Exception as e:
        logging.error(f"Error answering medical question: {e}")
        return {"error": str(e)}

# ------------------------------------------------------------------------------
# FLASK ROUTES
# ------------------------------------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate_text', methods=['POST'])
def generate_text_route():
    """
    Main route for generating text. If it looks like a question, 
    we handle it as factual QA; otherwise, open-ended text generation.
    """
    try:
        prompt = request.json.get('prompt', '').strip()
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        if requires_factual_answer(prompt):
            # Fact question → handle_fact_question
            answer = handle_fact_question(prompt)
            return jsonify({"result": answer, "confidence": 0.9})
        else:
            # Open-ended generation
            input_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
            response = generate_response(input_prompt)
            return jsonify({"result": response, "confidence": 0.9})

    except Exception as e:
        logging.error(f"Error generating text: {e}")
        return jsonify({"error": f"Error generating text: {e}"}), 500

@app.route('/analyze_image', methods=['POST'])
def analyze_image_route():
    try:
        image = request.files['image']
        result = analyze_image(image)
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error analyzing image: {e}")
        return jsonify({"error": f"Error analyzing image: {e}"}), 500

@app.route('/voice_command', methods=['POST'])
def voice_command():
    """
    Convert speech to text, then either do factual QA or open-ended generation.
    """
    try:
        audio = request.files['audio']
        text = speech_to_text(audio)
        # Decide if it's a question or not
        if requires_factual_answer(text):
            response = handle_fact_question(text)
        else:
            # Open-ended generation
            input_prompt = f"### Instruction:\n{text}\n\n### Response:\n"
            response = generate_response(input_prompt)

        speech_response = text_to_speech(response)
        return jsonify({'response': speech_response})

    except Exception as e:
        logging.error(f"Error processing voice command: {e}")
        return jsonify({"error": f"Error processing voice command: {e}"}), 500

@app.route('/business_info', methods=['POST'])
def business_info():
    try:
        query = request.json['query']
        location = request.json['location']
        info = get_business_info(query, location)
        return jsonify(info)
    except Exception as e:
        logging.error(f"Error fetching business info: {e}")
        return jsonify({"error": f"Error fetching business info: {e}"}), 500

@app.route('/read_document', methods=['POST'])
def read_document_route():
    global context_data
    try:
        document = request.files['document']
        content = read_document(document)
        if "error" in content:
            return jsonify({"error": f"Error reading document: {content['error']}"}), 500

        analysis = analyze_document(content)
        if "error" in analysis:
            return jsonify({"error": f"Error analyzing document: {analysis['error']}"}), 500

        context_data = content.get("content", "")
        logging.debug(f"context_data set to: {context_data[:200]}...")
        return jsonify({"analysis": analysis.get("analysis", {})})
    except Exception as e:
        logging.error(f"Error reading document: {e}")
        return jsonify({"error": f"Error reading document: {e}"}), 500

@app.route('/read_prescription', methods=['POST'])
def read_prescription_route():
    global prescription_context
    try:
        file = request.files.get('prescription')
        if not file:
            return jsonify({"error": "No file uploaded."}), 400
        content = read_prescription(file)
        if "error" in content:
            return jsonify({"error": content["error"]}), 500

        analysis = analyze_prescription(content)
        if "error" in analysis:
            return jsonify({"error": analysis["error"]}), 500

        prescription_context = content.get("content", "")
        return jsonify({"analysis": analysis.get("analysis", {})})
    except Exception as e:
        logging.error(f"Error reading prescription: {e}")
        return jsonify({"error": f"Error reading prescription: {e}"}), 500

@app.route('/ask_prescription_question', methods=['POST'])
def ask_prescription_question():
    global prescription_context
    try:
        question = request.json['question']
        if not prescription_context:
            return jsonify({"error": "No prescription context available. Please upload a prescription first."})

        # Use the fine-tuned LLaMA model to answer the question
        answer = answer_medical_question(question, prescription_context)
        return jsonify(answer)
    except Exception as e:
        logging.error(f"Error answering prescription question: {e}")
        return jsonify({"error": f"Error answering prescription question: {e}"}), 500

@app.route('/scrape', methods=['POST'])
async def scrape():
    global context_data
    try:
        url = request.json['url']
        data = await asyncio.to_thread(scrape_website, url)
        context_data = ' '.join(data['headings'] + data['paragraphs']) if 'headings' in data and 'paragraphs' in data else ''
        logging.debug(f"context_data set to: {context_data[:200]}...")
        return jsonify(data)
    except Exception as e:
        logging.error(f"Error scraping website: {e}")
        return jsonify({"error": f"Error scraping website: {e}"}), 500

@app.route('/ask_question', methods=['POST'])
def ask_question_route():
    global context_data
    try:
        question = request.json['question']
        if not context_data:
            return jsonify({"error": "No context available. Please upload a document or scrape a website first."})
        answer = generate_response(f"""
You are a helpful medical assistant and a top general doctor. If any health issues arise, you are the first person to consult for help. If you are asked questions, you need to provide concise and accurate answers.

Please answer the question below in 1-2 sentences, accurately and concisely.

Question: {question}
""".strip())
        return jsonify({"answer": answer})
    except Exception as e:
        logging.error(f"Error answering question: {e}")
        return jsonify({"error": f"Error answering question: {e}"}), 500

@app.route('/fact_question', methods=['POST'])
def fact_question_route():
    try:
        question = request.json['question']
        answer = handle_fact_question(question)
        return jsonify({"answer": answer})
    except Exception as e:
        logging.error(f"Error fetching fact: {e}")
        return jsonify({"error": f"Error fetching fact: {e}"}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        data = request.json
        question = data.get('question')
        answer = data.get('answer')
        helpful = data.get('helpful')
        feedback_entry = f"Question: {question}, Answer: {answer}, Helpful: {helpful}\n"

        with open(FEEDBACK_FILE, "a") as f:
            f.write(feedback_entry)

        return jsonify({"status": "Feedback received"})
    except Exception as e:
        logging.error(f"Error processing feedback: {e}")
        return jsonify({"error": f"Error processing feedback: {e}"}), 500

@app.route('/ask_question_audio', methods=['POST'])
def ask_question_audio():
    """
    Convert audio to text, then do QA with `context_data` if available.
    """
    global context_data
    try:
        audio = request.files['audio']
        question = speech_to_text(audio)
        if not context_data:
            return jsonify({"error": "No context available. Please upload a document or scrape a website first."})

        answer = generate_response(f"""
You are a helpful medical assistant and a top general doctor. If any health issues arise, you are the first person to consult for help. If you are asked questions, you need to provide concise and accurate answers.

Please answer the question below in 1-2 sentences, accurately and concisely.

Question: {question}
""".strip())
        return jsonify({"answer": answer, "question": question})
    except Exception as e:
        logging.error(f"Error processing audio question: {e}")
        return jsonify({"error": f"Error processing audio question: {e}"}), 500

@app.route('/ask_medical_question', methods=['POST'])
def ask_medical_question_route():
    global prescription_context
    try:
        question = request.json.get('question', '').strip()
        if not question:
            return jsonify({"error": "Question is required."}), 400

        # Correct the grammar by removing "the" before specific medical conditions if present
        # This regex can be expanded to handle more conditions as needed
        question = re.sub(r'\bthe\s+(cancer)\b', r'\1', question, flags=re.IGNORECASE)

        # Capitalize the first letter and ensure the question ends with a question mark
        question = question.capitalize()
        if not question.endswith('?'):
            question += '?'

        # Construct the refined prompt
        prompt = (
            f"""You are a helpful medical assistant and a top general doctor. If any health issues arise, you are the first person to consult for help. If you are asked questions, you need to provide concise and accurate answers.
            
Please answer the question below in 1-2 sentences, accurately and concisely.
            
Question: {question}
Answer:
            """.strip()
        )

        # Use the fine-tuned LLaMA model to answer the medical question
        answer = generate_response(prompt)
        return jsonify({"answer": answer})
    except Exception as e:
        logging.error(f"Error answering medical question: {e}")
        return jsonify({"error": f"Error answering medical question: {e}"}), 500

@app.route('/test_generate', methods=['POST'])
def test_generate():
    """
    Minimal Flask route to test text generation independently.
    """
    try:
        prompt = request.json.get('prompt', '').strip()
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        answer = generate_response(prompt)
        return jsonify({"answer": answer})
    except Exception as e:
        logging.error(f"Error in test_generate: {e}")
        return jsonify({"error": f"Error in test_generate: {e}"}), 500

if __name__ == '__main__':
    if not os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, 'w') as f:
            f.write("Feedback Log\n")
    # Run Flask in single-threaded mode for testing purposes
    app.run(debug=True, threaded=False)