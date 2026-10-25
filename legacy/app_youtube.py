import streamlit as st
from dotenv import load_dotenv
import os
import requests
import mysql.connector
from pathlib import Path
import time
import yt_dlp
import tempfile
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader

# Additional imports for conversational chain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document  # Import Document

from datetime import datetime
import uuid  # For generating unique session IDs
import re    # For regular expressions
import json  # For JSON parsing
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
api_token = os.getenv("ASSEMBLYAI_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key

# AssemblyAI base URL and headers
headers = {
    "authorization": api_token,
    "content-type": "application/json"
}

# MySQL Database Configuration
db_host = os.getenv('DB_HOST')
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_database = os.getenv('DB_DATABASE')

db_config = {
    'host': db_host,
    'user': db_user,
    'password': db_password,
    'database': db_database
}

def get_db_connection():
    """Create a connection to the MySQL database."""
    return mysql.connector.connect(**db_config)

# Database Functions
def store_transcription(session_id, transcription_text):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = """
    INSERT INTO transcription_sessions (session_id, transcription_text, created_at)
    VALUES (%s, %s, NOW())
    """
    cursor.execute(query, (session_id, transcription_text))
    conn.commit()
    conn.close()

def store_chat_message(session_id, role, content):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = """
    INSERT INTO chat_messages (session_id, role, content, created_at)
    VALUES (%s, %s, %s, NOW())
    """
    cursor.execute(query, (session_id, role, content))
    conn.commit()
    conn.close()

def get_transcription_sessions():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    query = """
    SELECT ts.session_id, ts.transcription_text, ts.created_at, COUNT(cm.id) as message_count
    FROM transcription_sessions ts
    LEFT JOIN chat_messages cm ON ts.session_id = cm.session_id
    GROUP BY ts.session_id
    ORDER BY ts.created_at DESC
    """
    cursor.execute(query)
    sessions = cursor.fetchall()
    conn.close()
    return sessions

def get_chat_messages(session_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    query = """
    SELECT role, content
    FROM chat_messages
    WHERE session_id = %s
    ORDER BY created_at ASC
    """
    cursor.execute(query, (session_id,))
    messages = cursor.fetchall()
    conn.close()
    return messages

# Function to download audio
def save_audio(url):
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "outtmpl": "%(title)s.%(ext)s",
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info_dict)
            if filename.endswith(".webm"):
                filename = filename.replace(".webm", ".mp3")
            elif filename.endswith(".m4a"):
                filename = filename.replace(".m4a", ".mp3")
            else:
                filename = Path(filename).stem + ".mp3"
            return filename
    except Exception as e:
        st.error(f"Error downloading or processing audio: {e}")
        return None

# Function to load configurations from JSON files
# Function to load configurations from JSON files
def load_configurations():
    try:
        with open('config/fillers.json', 'r') as f:
            fillers = json.load(f)['fillers']
        
        with open('config/abbreviations.json', 'r') as f:
            abbreviations = json.load(f)
        
        return fillers, abbreviations
    except Exception as e:
        st.error(f"Error loading configuration files: {e}")
        return [], {}

# Load configurations
fillers, abbreviations = load_configurations()

# Compile regex patterns
if fillers:
    filler_pattern = re.compile(r'\b(' + '|'.join(map(re.escape, fillers)) + r')\b', re.IGNORECASE)
else:
    filler_pattern = None

if abbreviations:
    abbr_pattern = re.compile(r'\b(' + '|'.join(map(re.escape, abbreviations.keys())) + r')\b', re.IGNORECASE)
else:
    abbr_pattern = None

# Function to clean transcription text
def clean_transcription_external_config(text):
    # Remove fillers
    if filler_pattern:
        text = filler_pattern.sub('', text)
        text = re.sub(' +', ' ', text)
    
    # Expand abbreviations
    if abbr_pattern:
        text = abbr_pattern.sub(lambda match: abbreviations.get(match.group(0).lower(), match.group(0)), text)
    
    return text.strip()


def reprocess_existing_transcriptions():
    sessions = get_transcription_sessions()
    for session in sessions:
        original_text = session['transcription_text']
        cleaned_text = clean_transcription_external_config(original_text)
        
        # Update the database with cleaned text
        conn = get_db_connection()
        cursor = conn.cursor()
        update_query = """
        UPDATE transcription_sessions
        SET transcription_text = %s
        WHERE session_id = %s
        """
        cursor.execute(update_query, (cleaned_text, session['session_id']))
        conn.commit()
        conn.close()
        
        # Update the transcription.txt file if necessary
        # (Assuming one transcription per session)
        transcription_file = f"docs/transcription_{session['session_id']}.txt"
        with open(transcription_file, "w") as file:
            file.write(cleaned_text)
    
    st.success("All existing transcriptions have been reprocessed and cleaned.")



# Function to transcribe audio
def assembly_trans(audio_filename):
    try:
        with open(audio_filename, "rb") as f:
            response = requests.post(
                "https://api.assemblyai.com/v2/upload",
                headers=headers,
                data=f
            )
        response.raise_for_status()
        upload_url = response.json()["upload_url"]

        data = {"audio_url": upload_url}
        response = requests.post(
            "https://api.assemblyai.com/v2/transcript",
            json=data,
            headers=headers
        )
        response.raise_for_status()
        transcript_id = response.json()["id"]

        polling_endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
        while True:
            transcript_result = requests.get(polling_endpoint, headers=headers)
            transcript_result.raise_for_status()
            transcript_json = transcript_result.json()
            if transcript_json["status"] == "completed":
                break
            elif transcript_json["status"] == "error":
                raise RuntimeError(f"Transcription failed: {transcript_json['error']}")
            else:
                time.sleep(3)

        transcription_text = transcript_json["text"]

        # Clean the transcription text
        cleaned_transcription = clean_transcription_external_config(transcription_text)

        os.makedirs("docs", exist_ok=True)
        with open("docs/transcription.txt", "w") as file:
            file.write(cleaned_transcription)

        return cleaned_transcription
    except Exception as e:
        st.error(f"Error in transcription: {e}")
        return None

# Function to create a Document with metadata
def create_document(content, source, session_id, role=None):
    metadata = {
        "source": source,
        "session_id": session_id,
        "timestamp": datetime.utcnow().isoformat()
    }
    if role:
        metadata["role"] = role
    return Document(page_content=content, metadata=metadata)

# Function to initialize vectorstore
def get_vectorstore():
    try:
        documents = []

        # Load transcription with metadata
        transcription_loader = TextLoader("docs/transcription.txt")
        transcription_docs = transcription_loader.load()
        for doc in transcription_docs:
            documents.append(create_document(
                content=doc.page_content,
                source="transcription",
                session_id=st.session_state.get("current_session", "default")
            ))

        # Load conversation messages with metadata
        if st.session_state["messages"]:
            for message in st.session_state["messages"]:
                role = message["role"]
                content = message["content"]
                documents.append(create_document(
                    content=f"{role}: {content}",
                    source="chat",
                    session_id=st.session_state.get("current_session", "default"),
                    role=role
                ))

        embeddings = OpenAIEmbeddings()

        # Define a persistent directory for the vector store
        vectorstore_dir = Path("vectorstore") / st.session_state.get("current_session", "default")
        os.makedirs(vectorstore_dir, exist_ok=True)

        vectorstore = Chroma.from_documents(
            documents, 
            embeddings, 
            persist_directory=str(vectorstore_dir)
        )
        vectorstore.persist()  # Ensure the vector store is saved

        # Customize retriever to filter by session_id
        retriever = vectorstore.as_retriever(search_kwargs={"filter": {"session_id": st.session_state.get("current_session", "default")}})

        return vectorstore, retriever
    except Exception as e:
        st.error(f"Error initializing vectorstore: {e}")
        return None, None

# Initialize Streamlit app
st.set_page_config(layout="centered", page_title="VoxAuraAI", page_icon="🔉")
st.title("VoxAuraAI")

USER_AVATAR = "👤"
BOT_AVATAR = "🤖"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "current_session" not in st.session_state:
    st.session_state["current_session"] = None
if "transcription" not in st.session_state:
    st.session_state["transcription"] = ""
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "retriever" not in st.session_state:
    st.session_state["retriever"] = None
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
if "qa_chain" not in st.session_state:
    st.session_state["qa_chain"] = None

# Function to initialize the conversational chain
def initialize_chain():
    if st.session_state["retriever"] and st.session_state["memory"]:
        if st.session_state["qa_chain"] is None:
            st.session_state["qa_chain"] = ConversationalRetrievalChain.from_llm(
                llm=ChatOpenAI(model_name="gpt-4", temperature=0),
                retriever=st.session_state["retriever"],
                memory=st.session_state["memory"]
            )

# Function to rebuild memory from messages
def add_document_to_vectorstore(content, source, role=None):
    if st.session_state["vectorstore"]:
        doc = create_document(
            content=content,
            source=source,
            session_id=st.session_state.get("current_session", "default"),
            role=role
        )
        # Add context threading to the document metadata
        if st.session_state["messages"]:
            prev_message = st.session_state["messages"][-1]
            doc.metadata["context"] = prev_message["content"]
        st.session_state["vectorstore"].add_documents([doc])
        st.session_state["vectorstore"].persist()


# Sidebar for transcription sessions
with st.sidebar:
    st.header("Transcription History")
    sessions = get_transcription_sessions()

    # Button to start a new transcription
    if st.button("New Transcription"):
        st.session_state["current_session"] = None
        st.session_state["messages"] = []
        st.session_state["transcription"] = ""
        st.session_state["vectorstore"] = None
        st.session_state["retriever"] = None
        st.session_state["memory"] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        st.session_state["qa_chain"] = None

    # Display previous sessions with formatted labels
    for session in sessions:
        # Format: Truncated transcription text (10 words max) followed by timestamp and message count
        truncated_text = ' '.join(session['transcription_text'].split()[:10])  # First 10 words
        created_at = session['created_at'].strftime("%Y-%m-%d %H:%M:%S")
        message_count = session['message_count']
        button_label = f"{truncated_text}... - {created_at} ({message_count} messages)"

        # Button to load a specific session
        if st.button(button_label, key=session["session_id"]):
            st.session_state["current_session"] = session["session_id"]
            st.session_state["messages"] = get_chat_messages(session["session_id"])
            st.session_state["transcription"] = session["transcription_text"]

            # Load transcription into vectorstore for Q&A
            os.makedirs("docs", exist_ok=True)
            with open("docs/transcription.txt", "w") as file:
                file.write(session["transcription_text"])
            vectorstore, retriever = get_vectorstore()
            st.session_state["vectorstore"] = vectorstore
            st.session_state["retriever"] = retriever

            if retriever:
                # Rebuild the conversation memory
                rebuild_memory()

                # Initialize the chain
                st.session_state["qa_chain"] = None
                initialize_chain()

# Input for the video URL
input_video = st.text_input("Enter the URL of the video")

# Process video and transcription
if input_video and st.button("Transcribe"):
    audio_filename = save_audio(input_video)
    if audio_filename:
        transcription_text = assembly_trans(audio_filename)
        if transcription_text:
            # Generate a unique session ID using UUID
            session_id = uuid.uuid4().hex[:16]
            # Store transcription in database
            store_transcription(session_id, transcription_text)
            st.session_state["current_session"] = session_id
            st.session_state["messages"] = []
            st.session_state["transcription"] = transcription_text
            st.success("Transcription completed and stored!")

            # Initialize vectorstore and retriever
            vectorstore, retriever = get_vectorstore()
            if vectorstore and retriever:
                st.session_state["vectorstore"] = vectorstore
                st.session_state["retriever"] = retriever

                # Reset memory and chain
                st.session_state["memory"] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                st.session_state["qa_chain"] = None
                initialize_chain()

# Display transcription
if st.session_state.get("transcription"):
    st.subheader("Transcription")
    st.write(st.session_state["transcription"])

# Display chat messages
for message in st.session_state["messages"]:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Function to add a document to the vectorstore with metadata
def add_document_to_vectorstore(content, source, role=None):
    if st.session_state["vectorstore"]:
        doc = create_document(
            content=content,
            source=source,
            session_id=st.session_state.get("current_session", "default"),
            role=role
        )
        st.session_state["vectorstore"].add_documents([doc])
        st.session_state["vectorstore"].persist()

if st.session_state.get("retriever") and st.session_state.get("qa_chain"):
    prompt = st.chat_input("Ask a question about the transcription")
    if prompt:
        # Append user message with context
        st.session_state["messages"].append({"role": "user", "content": prompt})
        store_chat_message(st.session_state["current_session"], "user", prompt)
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)

        # Add user message to vectorstore with metadata
        add_document_to_vectorstore(prompt, source="chat", role="user")

        # Generate assistant's response considering memory context
        response = st.session_state["qa_chain"]({"question": prompt})
        answer = response["answer"]

        # Append assistant's response with context
        st.session_state["messages"].append({"role": "assistant", "content": answer})
        store_chat_message(st.session_state["current_session"], "assistant", answer)
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            st.markdown(answer)

        # Add assistant response to vectorstore with metadata
        add_document_to_vectorstore(answer, source="chat", role="assistant")

        # Update the retriever
        st.session_state["retriever"] = st.session_state["vectorstore"].as_retriever()
