from transformers import pipeline

# Initialize the GPT-2 pipeline
gpt2_pipeline = pipeline('text-generation', model='meta-llama/Llama-3.2-1B')
qa_pipeline = pipeline('question-answering' , model='meta-llama/Llama-3.2-1B')

def generate_text(prompt):
    result = gpt2_pipeline(prompt, max_length=100, num_return_sequences=1)
    return result[0]['generated_text']

def answer_question(question, context):
    result = qa_pipeline(question=question, context=context)
    return result['answer']
