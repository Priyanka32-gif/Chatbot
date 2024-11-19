#import the necessary libraries
import openai
import os
import re
import langchain
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
import nltk
import tiktoken
import numpy as np
import faiss
from pymongo import MongoClient
from flask import Flask, render_template, request, jsonify
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
import dateparser
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Now retrieve the API key
openai.api_key = os.getenv("OPENAI_API_KEY")

client = MongoClient(os.getenv("MONGO_URL"))

db = client['User_details']

collection = db['Form detail']


#using flask api
app = Flask(__name__)

from flask_cors import CORS
CORS(app)


@app.route('/')
def index():
    return render_template('UI.html') #user interface


# Open the PDF file
path = "History_Ancient_Medieval_Nepal.pdf"

#load the document
def read_doc(path):
    """Load and return documents from a PDF file."""
    file_loader = PyPDFLoader(path)
    documents = file_loader.load()
    return documents


documents = read_doc(path)

# Extract page content from the documents
docs_content = [Document(page_content=doc.page_content) for doc in documents]

# divide the document content into chunks
def chunk_data(documents, sentences_per_chunk=3):
    """Chunk documents into smaller chunks of sentences."""
    document_chunks = []

    for doc in documents:
        sentences = nltk.sent_tokenize(doc.page_content)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

        for i in range(0, len(sentences), sentences_per_chunk):
            chunk = " ".join(sentences[i:i + sentences_per_chunk])
            document_chunks.append(chunk)

    return document_chunks


docs_chunks_list = chunk_data(documents=docs_content)

#clean the content of the chunks
def clean_chunk_list(chunk):
    """Clean and tokenize chunk text."""
    text = chunk.page_content if isinstance(chunk, Document) else chunk
    cleaned_text = re.sub(r'\n+', ' ', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    tokens = cleaned_text.split(' ')
    tokens = [token for token in tokens if token.strip()]
    formatted_paragraph = ' '.join(tokens)
    return formatted_paragraph


clean_chunks = [clean_chunk_list(chunk) for chunk in docs_chunks_list]

#truncate the text to not exceed the limit
def truncate_text(text, max_tokens=1000):
    """Truncate text if it exceeds the max token limit."""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return tokenizer.decode(tokens)


# function to get the embedding for the chunks data
def get_embedding_for_chunk(chunk):
    """Generate embedding for a single chunk."""
    truncated_chunk = truncate_text(chunk, max_tokens=1000)
    try:
        response = openai.Embedding.create(input=truncated_chunk, model="text-embedding-ada-002")
        return response['data'][0]['embedding']
    except openai.error.OpenAIError as e:
        print(f"Error generating embedding: {e}")
        return None

# enumerate the clean text to get the embeddings
chunk_embeddings = []
for i, chunk in enumerate(clean_chunks):
    try:
        embedding = get_embedding_for_chunk(chunk)
        chunk_embeddings.append(embedding)
    except openai.error.InvalidRequestError as e:
        print(f"Error generating embedding for chunk {i + 1}: {e}")


chunk_embeddings = np.array(chunk_embeddings).astype('float32')

d = chunk_embeddings.shape[1]
print(f"Dimension of embeddings: {d}")

#use faiss index for vector database
index = faiss.IndexFlatL2(d)
print(f"FAISS Index Type: {type(index)}")

index.add(chunk_embeddings)
print(f"Number of embeddings in index: {index.ntotal}")

# function for similarity search and it is quality search
def embed_and_search(text, index, k=5):
    """Embed the input text and search in the FAISS index."""
    embedding = get_embedding_for_chunk(text)
    embedding = np.array([embedding]).astype('float32')
    distances, indices = index.search(embedding, k)

    contents = []
    for idx in indices[0]:
        contents.append(clean_chunks[idx])

    return contents


#query_text = "Tell me about Soma Dynasty"
#content = embed_and_search(query_text, index)

# function to retrive the content from the document to answer the user query
def document_retrive_convo(user_question, retrieved_content, chat_history):
    """Generate a chatbot response based on the retrieved content and chat history."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    "You are an intelligent assistant that answers the user's questions based on their query, "
                    "retrieved content, and chat history. Respond in a conversational tone and naturally. "
                )},
                {"role": "user", "content": (
                    f"The user asked: '{user_question}'. Here is the content related to the question:\n\n"
                    f"{retrieved_content}\n\n"
                    f"Chat history:\n\n{chat_history}\n\n"
            
                )}
            ],
            max_tokens=150,
            temperature=0.3,
        )
        summary = response.choices[0].message['content'].strip()
        return summary

    except Exception as e:
        print(f"OpenAI API Error: {e}")
        return "Sorry, I couldn't process that request at the moment."


def validate_email(email):
    """Validate email format."""
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)


def validate_phone(phone):
    return bool(re.match(r"^(97|98)\d{8}$", phone))

def parse_date(date_text):
    """Parse a date in natural language to a standard format."""
    return dateparser.parse(date_text).strftime("%Y-%m-%d") if dateparser.parse(date_text) else None


def book_appointment(name, phone, email, date):
    return f"Appointment booked successfully for {name} on {date}. Confirmation sent to {email}."


tools = [
    Tool(name="ValidateEmail", func=validate_email, description="Validate email address."),
    Tool(name="ValidatePhone", func=validate_phone, description="Validate phone number."),
    Tool(name="ParseDate", func=parse_date, description="Parse natural language dates into YYYY-MM-DD."),
    Tool(name="BookAppointment", func=book_appointment, description="Book an appointment with user details.")
]

history = []  


@app.route('/chat', methods=['POST'])
def chat():

    data = request.json
    user_input = data.get('message', '').strip()

    # If the user hasn't sent a message yet, greet them
    if not user_input:
        greeting_message = (
            "Hello! I’m your assistant. How can I help you today? "
            "You can ask me questions for Nepal's history book, or ask to save contact info, or book an appointment!"
        )
        history.append({'user': None, 'bot': greeting_message})
        return jsonify({'reply': greeting_message})

    if 'call' in user_input:
        return handle_user_information(user_input, save_only=True)
    elif "book" or "appointment" in user_input:
        return handle_user_information(user_input, save_only=False)
    

    relevant_content = embed_and_search(user_input, index)

    formatted_history = "\n\n".join([f"User: {entry['user']}\nBot: {entry['bot']}" for entry in history if entry['user']])

    bot_reply = document_retrive_convo(user_input, relevant_content, formatted_history)

    history.append({'user': user_input, 'bot': bot_reply})
    history[:] = history[-2:]

    return jsonify({'reply': bot_reply})

user_info = {}

def handle_user_information(user_input,save_only):
    """Handle user information collection during booking or appointment process."""
    count = 0
    if count == 0:
        count +=1
        return jsonify({'reply': "Got it! Can I have your name first?"})

    if count == 1:
        user_info["name"] = user_input
        count += 1
        return jsonify({'reply': f"Thanks, {user_info['name']}! May I have your phone number now?"})

    elif count == 2:
        if not validate_phone(user_input):
            return jsonify({'reply': "That doesn't seem like a valid Nepali phone number. Please provide a valid number."})
        user_info["phone"] = user_input
        count += 1
        return jsonify({'reply': "Great! Could you share your email address?"})


    elif count == 3:
        if not validate_email(user_input):
            return jsonify({'reply': "That doesn't seem like a valid email address. Please provide a valid email."})
        user_info["email"] = user_input



        existing_user = collection.find_one({"phone": user_info["phone"], "email": user_info["email"]})

        if existing_user:
            response = (
                f"Thank you, {user_info['name']}! I see your details are already in our database. "
                "We'll use them to assist you better!"
            )
        else:
            collection.insert_one({
                "name": user_info["name"],
                "phone": user_info["phone"],
                "email": user_info["email"]
            })

        if save_only:
            response = f"Thank you, {user_info['name']}! Your details have been saved, and we'll call you soon."
            user_info.clear()
            return jsonify({'reply': response})
    
        else:
            count += 1
            return jsonify({'reply': "Th-99anks! What date would you prefer for your appointment?"})

    elif count == 4:
        appointment_date = parse_date(user_input)
        if not appointment_date:
            return jsonify({'reply': "I couldn't understand the date. Please provide a valid date (e.g., YYYY-MM-DD or 'next Monday')."})
        user_info["date"] = appointment_date
        response = book_appointment(user_info["name"], user_info["phone"], user_info["email"], user_info["date"])

        user_info.clear()
        
        return jsonify({'reply': response})

if __name__ == '__main__':
    app.run(debug=True)
