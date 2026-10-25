import streamlit as st
import os
import requests
import time
import uuid
import re
import json
import logging
import mysql.connector
from mysql.connector import Error
from pathlib import Path
from dotenv import load_dotenv

# For st_audiorec (microphone recording):
import numpy as np
from scipy.io.wavfile import write
from st_audiorec import st_audiorec

# For YouTube audio download:
import yt_dlp

# LangChain imports
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# --------------------------- #
#        ENV & LOGGING        #
# --------------------------- #
logging.basicConfig(level=logging.INFO)
load_dotenv()

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# --------------------------- #
#   ASSEMBLYAI CONFIG         #
# --------------------------- #
assembly_headers = {
    "authorization": ASSEMBLYAI_API_KEY,
    "content-type": "application/json"
}
assembly_base_url = "https://api.assemblyai.com/v2"

# --------------------------- #
#       MYSQL CONFIG          #
# --------------------------- #
db_config = {
    'host': os.getenv("DB_HOST"),
    'user': os.getenv("DB_USER"),
    'password': os.getenv("DB_PASSWORD"),
    'database': os.getenv("DB_DATABASE")
}

def get_db_connection():
    """Create a connection to the MySQL database."""
    return mysql.connector.connect(**db_config)

# --------------------------- #
#   DATABASE HELPER METHODS   #
# --------------------------- #
def store_transcription(session_id, transcription_text):
    """Store the transcription text in the database under a given session_id."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        query = """
            INSERT INTO transcription_sessions (session_id, transcription_text, created_at)
            VALUES (%s, %s, NOW())
        """
        cursor.execute(query, (session_id, transcription_text))
        conn.commit()
    except Error as e:
        st.error(f"Database error (store_transcription): {e}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def store_chat_message(session_id, role, content):
    """Store a single chat message in the database, referencing the session_id."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        query = """
            INSERT INTO chat_messages (session_id, role, content, created_at)
            VALUES (%s, %s, %s, NOW())
        """
        cursor.execute(query, (session_id, role, content))
        conn.commit()
    except Error as e:
        st.error(f"Database error (store_chat_message): {e}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def get_transcription_sessions():
    """
    Retrieve all transcription sessions with:
    - session_id
    - transcription_text
    - created_at
    - message_count (# of chat messages in that session)
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        query = """
            SELECT ts.session_id, ts.transcription_text, ts.created_at, 
                   COUNT(cm.id) AS message_count
            FROM transcription_sessions ts
            LEFT JOIN chat_messages cm ON ts.session_id = cm.session_id
            GROUP BY ts.session_id
            ORDER BY ts.created_at DESC
        """
        cursor.execute(query)
        sessions = cursor.fetchall()
        return sessions
    except Error as e:
        st.error(f"Database error (get_transcription_sessions): {e}")
        return []
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def get_chat_messages(session_id):
    """Retrieve all chat messages associated with a session_id."""
    try:
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
        return messages
    except Error as e:
        st.error(f"Database error (get_chat_messages): {e}")
        return []
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# --------------------------- #
#     CONFIG FOR CLEANING     #
# --------------------------- #
def load_configurations():
    """Load filler words and abbreviations from config files."""
    try:
        with open('config/fillers.json', 'r') as f:
            fillers = json.load(f)['fillers']
        
        with open('config/abbreviations.json', 'r') as f:
            abbreviations = json.load(f)
        
        return fillers, abbreviations
    except Exception as e:
        st.error(f"Error loading configuration files: {e}")
        return [], {}

fillers, abbreviations = load_configurations()

# Compile regex patterns for filler removal & abbreviations
if fillers:
    filler_pattern = re.compile(r'\b(' + '|'.join(map(re.escape, fillers)) + r')\b', re.IGNORECASE)
else:
    filler_pattern = None

if abbreviations:
    abbr_pattern = re.compile(r'\b(' + '|'.join(map(re.escape, abbreviations.keys())) + r')\b', re.IGNORECASE)
else:
    abbr_pattern = None

def clean_transcription_external_config(text):
    """Remove filler words and expand abbreviations in the given text."""
    # Remove fillers
    if filler_pattern:
        text = filler_pattern.sub('', text)
        text = re.sub(' +', ' ', text)  # collapse extra spaces
    
    # Expand abbreviations
    if abbr_pattern:
        text = abbr_pattern.sub(lambda match: abbreviations.get(match.group(0).lower(), match.group(0)), text)
    
    return text.strip()

# --------------------------- #
#  AUDIO TRANSCRIPTION LOGIC  #
# --------------------------- #
def transcribe_via_assemblyai(audio_filename):
    """Transcribe a local audio file via AssemblyAI, with cleaning."""
    try:
        # 1) Upload
        with open(audio_filename, "rb") as f:
            response = requests.post(
                assembly_base_url + "/upload",
                headers=assembly_headers,
                data=f
            )
        response.raise_for_status()
        upload_url = response.json()["upload_url"]

        # 2) Request transcription
        data = {"audio_url": upload_url}
        response = requests.post(
            assembly_base_url + "/transcript",
            json=data,
            headers=assembly_headers
        )
        response.raise_for_status()
        transcript_id = response.json()["id"]

        # 3) Poll for result
        polling_endpoint = f"{assembly_base_url}/transcript/{transcript_id}"
        while True:
            transcript_result = requests.get(polling_endpoint, headers=assembly_headers)
            transcript_result.raise_for_status()
            transcript_json = transcript_result.json()
            if transcript_json["status"] == "completed":
                break
            elif transcript_json["status"] == "error":
                raise RuntimeError(f"Transcription failed: {transcript_json['error']}")
            else:
                time.sleep(3)

        transcription_text = transcript_json["text"]

        # 4) Clean text
        cleaned_text = clean_transcription_external_config(transcription_text)

        # Save locally
        os.makedirs("docs", exist_ok=True)
        with open("docs/transcription.txt", "w") as file:
            file.write(cleaned_text)

        return cleaned_text

    except Exception as e:
        st.error(f"Error in transcription: {e}")
        return None

def download_audio_youtube(youtube_url):
    """
    Download audio from a YouTube link using yt_dlp.
    Returns the local MP3 filename or None if error.
    """
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "outtmpl": "%(title)s.%(ext)s"
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=True)
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

# --------------------------- #
#  LANGCHAIN + VECTOR STORE   #
# --------------------------- #

def create_document(content, source, session_id, role=None):
    """Create a Document object with metadata for the given content."""
    metadata = {
        "source": source,
        "session_id": session_id,
        "timestamp": time.time(),
    }
    if role:
        metadata["role"] = role
    return Document(page_content=content, metadata=metadata)

def build_vectorstore():
    """
    Build (or rebuild) the vector store from the current session's transcription
    plus the chat messages in st.session_state["messages"].
    """
    try:
        documents = []

        # 1) Load transcription from docs/transcription.txt
        if os.path.exists("docs/transcription.txt"):
            loader = TextLoader("docs/transcription.txt")
            txt_docs = loader.load()
            for doc in txt_docs:
                doc_with_meta = create_document(
                    content=doc.page_content,
                    source="transcription",
                    session_id=st.session_state["current_session"]
                )
                documents.append(doc_with_meta)

        # 2) Load chat messages from session_state
        for msg in st.session_state["messages"]:
            role = msg["role"]
            content = msg["content"]
            doc_with_meta = create_document(
                content=content,
                source="chat",
                session_id=st.session_state["current_session"],
                role=role
            )
            documents.append(doc_with_meta)

        embeddings = OpenAIEmbeddings()

        # Directory for the vector store (per session)
        persist_dir = Path("vectorstore") / st.session_state["current_session"]
        os.makedirs(persist_dir, exist_ok=True)

        # Build new store
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=str(persist_dir)
        )
        vectorstore.persist()

        # A retriever that filters by session_id
        retriever = vectorstore.as_retriever(
            search_kwargs={"filter": {"session_id": st.session_state["current_session"]}}
        )
        return vectorstore, retriever

    except Exception as e:
        st.error(f"Error building vectorstore: {e}")
        return None, None

def initialize_qa_chain():
    """Initialize the ConversationalRetrievalChain if needed."""
    if not st.session_state.get("qa_chain"):
        if st.session_state["retriever"] and st.session_state["memory"]:
            st.session_state["qa_chain"] = ConversationalRetrievalChain.from_llm(
                llm=ChatOpenAI(model_name="gpt-4", temperature=0),
                retriever=st.session_state["retriever"],
                memory=st.session_state["memory"]
            )

def add_document_to_vectorstore(content, source, role=None):
    """
    Add a single new Document to the existing vectorstore for the current session.
    """
    if st.session_state.get("vectorstore"):
        doc = create_document(
            content=content,
            source=source,
            session_id=st.session_state["current_session"],
            role=role
        )
        st.session_state["vectorstore"].add_documents([doc])
        st.session_state["vectorstore"].persist()

# --------------------------- #
#    META QUESTION HANDLER    #
# --------------------------- #
def handle_meta_question(user_input):
    """
    Check if user_input is a meta-question (e.g., "What was my first question?")
    and return an appropriate answer from st.session_state["user_questions"] if found.
    """
    lower_input = user_input.lower().strip()

    if "first question" in lower_input:
        if len(st.session_state["user_questions"]) > 0:
            return f"Your first question was: '{st.session_state['user_questions'][0]}'"
        else:
            return "You haven't asked any question yet."

    if "previous question" in lower_input or "last question" in lower_input:
        if len(st.session_state["user_questions"]) > 1:
            return f"Your previous question was: '{st.session_state['user_questions'][-2]}'"
        else:
            return "You don't have a previous question before this one."

    # Not a recognized meta-question
    return None

# --------------------------- #
#       STREAMLIT APP         #
# --------------------------- #
st.set_page_config(layout="centered", page_title="VoxAuraAI", page_icon="🔉")
st.title("VoxAuraAI: Combined Interface")

USER_AVATAR = "👤"
BOT_AVATAR  = "🤖"

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
if "user_questions" not in st.session_state:
    st.session_state["user_questions"] = []
if "audio_transcribed" not in st.session_state:
    st.session_state["audio_transcribed"] = False  # for microphone approach or file approach

# --------------------------- #
#         SIDEBAR             #
# --------------------------- #
with st.sidebar:
    st.header("Transcription Sessions")

    # Button to start a new blank session
    if st.button("New Session"):
        st.session_state["current_session"] = None
        st.session_state["messages"] = []
        st.session_state["transcription"] = ""
        st.session_state["vectorstore"] = None
        st.session_state["retriever"] = None
        st.session_state["memory"] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        st.session_state["qa_chain"] = None
        st.session_state["user_questions"] = []
        st.session_state["audio_transcribed"] = False

    sessions = get_transcription_sessions()
    for session in sessions:
        truncated_text = " ".join(session["transcription_text"].split()[:10])
        message_count  = session["message_count"]
        created_at     = session["created_at"].strftime("%Y-%m-%d %H:%M:%S")
        label = f"{truncated_text}... | {created_at} ({message_count} msgs)"

        if st.button(label, key=session["session_id"]):
            st.session_state["current_session"] = session["session_id"]
            st.session_state["transcription"]   = session["transcription_text"]
            st.session_state["messages"]        = get_chat_messages(session["session_id"])
            st.session_state["user_questions"]  = []

            # Write to docs/transcription.txt
            os.makedirs("docs", exist_ok=True)
            with open("docs/transcription.txt", "w") as file:
                file.write(session["transcription_text"])

            # Build vectorstore & retriever
            vs, rtr = build_vectorstore()
            st.session_state["vectorstore"] = vs
            st.session_state["retriever"]   = rtr

            # Reset memory & chain
            st.session_state["memory"]   = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            st.session_state["qa_chain"] = None
            initialize_qa_chain()

            st.session_state["audio_transcribed"] = True

# --------------------------- #
#  SELECT AUDIO INPUT METHOD  #
# --------------------------- #
st.subheader("Choose Your Audio Input Method")
input_method = st.radio(
    "Select an option:",
    ("Microphone", "File Upload", "YouTube Link")
)

# --------------- (1) MICROPHONE ---------------
if input_method == "Microphone":
    st.write("Press 'Start Recording', then press 'Stop' when done.")
    audio_data = st_audiorec()

    if not st.session_state["audio_transcribed"]:
        if isinstance(audio_data, dict) and audio_data.get("arr") is not None:
            audio_np = np.array(audio_data["arr"]).astype(np.float32)
            mic_filename = "mic_recorded_audio.wav"
            write(mic_filename, 44100, audio_np)
            st.success("Audio recorded. Transcribing...")

            # Transcribe
            transcription_text = transcribe_via_assemblyai(mic_filename)
            if transcription_text:
                st.success("Transcription completed.")
                st.write(transcription_text)

                if not st.session_state["current_session"]:
                    st.session_state["current_session"] = uuid.uuid4().hex[:16]

                store_transcription(st.session_state["current_session"], transcription_text)
                st.session_state["transcription"] = transcription_text

                vs, rtr = build_vectorstore()
                st.session_state["vectorstore"] = vs
                st.session_state["retriever"]   = rtr

                st.session_state["memory"]   = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                st.session_state["qa_chain"] = None
                initialize_qa_chain()

                st.session_state["audio_transcribed"] = True

        elif isinstance(audio_data, bytes):
            mic_filename = "mic_recorded_audio.wav"
            with open(mic_filename, "wb") as f:
                f.write(audio_data)
            st.success("Audio recorded. Transcribing...")

            transcription_text = transcribe_via_assemblyai(mic_filename)
            if transcription_text:
                st.success("Transcription completed.")
                st.write(transcription_text)

                if not st.session_state["current_session"]:
                    st.session_state["current_session"] = uuid.uuid4().hex[:16]

                store_transcription(st.session_state["current_session"], transcription_text)
                st.session_state["transcription"] = transcription_text

                vs, rtr = build_vectorstore()
                st.session_state["vectorstore"] = vs
                st.session_state["retriever"]   = rtr

                st.session_state["memory"]   = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                st.session_state["qa_chain"] = None
                initialize_qa_chain()

                st.session_state["audio_transcribed"] = True

        else:
            st.info("Record audio using the mic widget above.")
    else:
        # If we've already transcribed, just show it
        if st.session_state["transcription"]:
            st.write("Current Transcription:")
            st.write(st.session_state["transcription"])

# --------------- (2) FILE UPLOAD ---------------
elif input_method == "File Upload":
    uploaded_audio = st.file_uploader("Upload your audio file (mp3, wav, m4a)", type=["mp3", "wav", "m4a"])
    if uploaded_audio is not None and st.button("Transcribe File"):
        temp_file_name = f"uploaded_{uuid.uuid4().hex[:8]}.wav"
        with open(temp_file_name, "wb") as f:
            f.write(uploaded_audio.read())

        transcription_text = transcribe_via_assemblyai(temp_file_name)
        if transcription_text:
            st.success("Transcription completed.")
            st.write(transcription_text)

            if not st.session_state["current_session"]:
                st.session_state["current_session"] = uuid.uuid4().hex[:16]

            store_transcription(st.session_state["current_session"], transcription_text)
            st.session_state["transcription"] = transcription_text

            vs, rtr = build_vectorstore()
            st.session_state["vectorstore"] = vs
            st.session_state["retriever"]   = rtr

            st.session_state["memory"]   = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            st.session_state["qa_chain"] = None
            initialize_qa_chain()

# --------------- (3) YOUTUBE LINK ---------------
elif input_method == "YouTube Link":
    youtube_url = st.text_input("Paste YouTube URL here")
    if youtube_url and st.button("Download & Transcribe"):
        audio_filename = download_audio_youtube(youtube_url)
        if audio_filename:
            transcription_text = transcribe_via_assemblyai(audio_filename)
            if transcription_text:
                st.success("Transcription completed.")
                st.write(transcription_text)

                if not st.session_state["current_session"]:
                    st.session_state["current_session"] = uuid.uuid4().hex[:16]

                store_transcription(st.session_state["current_session"], transcription_text)
                st.session_state["transcription"] = transcription_text

                vs, rtr = build_vectorstore()
                st.session_state["vectorstore"] = vs
                st.session_state["retriever"]   = rtr

                st.session_state["memory"]   = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                st.session_state["qa_chain"] = None
                initialize_qa_chain()

# --------------------------- #
#       SHOW TRANSCRIPTION    #
# --------------------------- #
if st.session_state.get("transcription"):
    st.subheader("Current Transcription")
    st.write(st.session_state["transcription"])

# --------------------------- #
#       SHOW CHAT HISTORY     #
# --------------------------- #
for msg in st.session_state["messages"]:
    avatar = USER_AVATAR if msg["role"] == "user" else BOT_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# --------------------------- #
#       MAIN CHAT INPUT       #
# --------------------------- #
if st.session_state.get("retriever") and st.session_state.get("qa_chain"):
    prompt = st.chat_input("Ask a question about the transcription...")
    if prompt:
        # Keep track of user questions (for meta-questions)
        st.session_state["user_questions"].append(prompt)

        # 1) Check if meta-question
        meta_answer = handle_meta_question(prompt)
        if meta_answer:
            # It's a meta-answer. Respond directly.
            st.session_state["messages"].append({"role": "assistant", "content": meta_answer})
            store_chat_message(st.session_state["current_session"], "assistant", meta_answer)
            with st.chat_message("assistant", avatar=BOT_AVATAR):
                st.markdown(meta_answer)
        else:
            # Normal question flow
            st.session_state["messages"].append({"role": "user", "content": prompt})
            store_chat_message(st.session_state["current_session"], "user", prompt)
            with st.chat_message("user", avatar=USER_AVATAR):
                st.markdown(prompt)

            # Add user message to vectorstore
            add_document_to_vectorstore(prompt, source="chat", role="user")

            # LLM response
            response_dict = st.session_state["qa_chain"]({"question": prompt})
            answer = response_dict["answer"]

            st.session_state["messages"].append({"role": "assistant", "content": answer})
            store_chat_message(st.session_state["current_session"], "assistant", answer)
            with st.+("assistant", avatar=BOT_AVATAR):
                st.markdown(answer)

            # Add assistant response to vectorstore
            add_document_to_vectorstore(answer, source="chat", role="assistant")

            # Update retriever
            st.session_state["retriever"] = st.session_state["vectorstore"].as_retriever(
                search_kwargs={"filter": {"session_id": st.session_state["current_session"]}}
            )
