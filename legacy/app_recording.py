import streamlit as st
import os
import requests
import time
import uuid
import mysql.connector
from mysql.connector import Error
import shelve
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.schema import Document

# For the Conversational Chain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# For audio recording
import numpy as np
from scipy.io.wavfile import write
from st_audiorec import st_audiorec

# --------------------------- #
#         ENV SETUP           #
# --------------------------- #
load_dotenv()

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
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
    """
    Store the transcription text in the database under a given session_id.
    """
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
    """
    Store a single chat message in the database, referencing the session_id.
    """
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
    Retrieve all transcription sessions with some basic info:
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
    """
    Retrieve all chat messages associated with a session_id.
    """
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
#  AUDIO TRANSCRIPTION LOGIC  #
# --------------------------- #
def assembly_trans(audio_filename):
    """
    Transcribe an audio file via AssemblyAI.
    """
    try:
        # 1. Upload audio
        with open(audio_filename, "rb") as f:
            response = requests.post(
                assembly_base_url + "/upload",
                headers=assembly_headers,
                data=f
            )
        response.raise_for_status()
        upload_url = response.json()["upload_url"]

        # 2. Request transcription
        data = {"audio_url": upload_url}
        response = requests.post(
            assembly_base_url + "/transcript",
            json=data,
            headers=assembly_headers
        )
        response.raise_for_status()
        transcript_id = response.json()["id"]

        # 3. Poll for transcript
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

        # 4. Write transcription locally
        os.makedirs("docs", exist_ok=True)
        with open("docs/transcription.txt", "w") as file:
            file.write(transcription_text)

        return transcription_text

    except Exception as e:
        st.error(f"Error in transcription: {e}")
        return None

# --------------------------- #
#  LANGCHAIN + VECTOR STORE   #
# --------------------------- #

def create_document(content, source, session_id, role=None):
    """
    Create a LangChain Document object with metadata for the given content.
    """
    metadata = {
        "source": source,
        "session_id": session_id
    }
    if role:
        metadata["role"] = role
    return Document(page_content=content, metadata=metadata)

def build_vectorstore():
    """
    Build (or rebuild) the vectorstore from the current session's transcription
    AND the stored chat history in st.session_state["messages"].
    """
    try:
        documents = []

        # 1) Add transcription from docs/transcription.txt if it exists
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

        # 2) Add chat messages from the session_state
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

        # Persist directory per session
        persist_dir = Path("vectorstore") / st.session_state["current_session"]
        os.makedirs(persist_dir, exist_ok=True)

        # Build a new vectorstore for just this session's documents
        vectorstore = Chroma.from_documents(
            documents,
            embeddings,
            persist_directory=str(persist_dir)
        )
        vectorstore.persist()

        # Return a retriever that filters on session_id
        retriever = vectorstore.as_retriever(
            search_kwargs={"filter": {"session_id": st.session_state["current_session"]}}
        )

        return vectorstore, retriever
    except Exception as e:
        st.error(f"Error building vectorstore: {e}")
        return None, None

def initialize_qa_chain():
    """
    Initialize the ConversationalRetrievalChain if not already present.
    """
    if not st.session_state.get("qa_chain"):
        if st.session_state["retriever"] and st.session_state["memory"]:
            st.session_state["qa_chain"] = ConversationalRetrievalChain.from_llm(
                llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
                retriever=st.session_state["retriever"],
                memory=st.session_state["memory"],
            )

def add_document_to_vectorstore(content, source, role=None):
    """
    Add a single new Document to the existing vectorstore for the current session.
    """
    doc = create_document(
        content=content,
        source=source,
        session_id=st.session_state["current_session"],
        role=role
    )
    if st.session_state.get("vectorstore"):
        st.session_state["vectorstore"].add_documents([doc])
        st.session_state["vectorstore"].persist()

# --------------------------- #
#     STREAMLIT APP LAYOUT    #
# --------------------------- #
st.set_page_config(layout="centered", page_title="VoxAuraAI", page_icon="🔉")
st.title("VoxAuraAI")

USER_AVATAR = "👤"
BOT_AVATAR = "🤖"

# --------------------------- #
#   SESSION STATE INIT        #
# --------------------------- #
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
    st.session_state["memory"] = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

if "qa_chain" not in st.session_state:
    st.session_state["qa_chain"] = None

# A boolean/flag to let us know we've done the transcription once
if "audio_transcribed" not in st.session_state:
    st.session_state["audio_transcribed"] = False


# --------------------------- #
#        SIDEBAR              #
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
        st.session_state["memory"] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        st.session_state["qa_chain"] = None
        st.session_state["audio_transcribed"] = False

    sessions = get_transcription_sessions()
    for session in sessions:
        truncated_text = " ".join(session["transcription_text"].split()[:10])
        message_count = session["message_count"]
        created_at = session["created_at"].strftime("%Y-%m-%d %H:%M:%S")
        label = f"{truncated_text}... | {created_at} ({message_count} msgs)"

        if st.button(label, key=session["session_id"]):
            # Load that session's data
            st.session_state["current_session"] = session["session_id"]
            st.session_state["transcription"] = session["transcription_text"]
            st.session_state["messages"] = get_chat_messages(session["session_id"])

            # Write session transcription to docs/transcription.txt
            os.makedirs("docs", exist_ok=True)
            with open("docs/transcription.txt", "w") as file:
                file.write(session["transcription_text"])

            # Build vectorstore & retriever
            vs, rtr = build_vectorstore()
            st.session_state["vectorstore"] = vs
            st.session_state["retriever"] = rtr

            # Reset memory and chain
            st.session_state["memory"] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            st.session_state["qa_chain"] = None
            initialize_qa_chain()

            # Mark that we've loaded an existing session
            st.session_state["audio_transcribed"] = True

# --------------------------- #
#  RECORD AUDIO + TRANSCRIBE  #
# --------------------------- #
st.header("Record Your Audio")
audio_data = st_audiorec()

if not st.session_state["audio_transcribed"]:  
    # Only transcribe if we haven't done it yet in this session
    if isinstance(audio_data, dict) and audio_data.get("arr") is not None:
        # st_audiorec returns a dict with "arr" key containing float32 audio data
        audio_np = np.array(audio_data["arr"]).astype(np.float32)
        audio_filename = "recorded_audio.wav"
        write(audio_filename, 44100, audio_np)

        st.success("Audio recording saved. Transcribing...")

        # 1) Transcribe
        transcription_text = assembly_trans(audio_filename)
        if transcription_text:
            st.success("Transcription completed.")
            st.write(transcription_text)

            # 2) Create a new session_id if we don't have one
            if not st.session_state["current_session"]:
                st.session_state["current_session"] = uuid.uuid4().hex[:16]

            # 3) Store transcription in MySQL
            store_transcription(st.session_state["current_session"], transcription_text)
            st.session_state["transcription"] = transcription_text

            # 4) Build vectorstore & retriever
            vs, rtr = build_vectorstore()
            st.session_state["vectorstore"] = vs
            st.session_state["retriever"] = rtr

            # 5) Reset memory & chain
            st.session_state["memory"] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            st.session_state["qa_chain"] = None
            initialize_qa_chain()

            # Mark transcription done
            st.session_state["audio_transcribed"] = True

    elif isinstance(audio_data, bytes):
        # If st_audiorec returns raw bytes
        audio_filename = "recorded_audio.wav"
        with open(audio_filename, "wb") as f:
            f.write(audio_data)

        st.success("Audio recording saved. Transcribing...")

        # 1) Transcribe
        transcription_text = assembly_trans(audio_filename)
        if transcription_text:
            st.success("Transcription completed.")
            st.write(transcription_text)

            # 2) Create a new session_id if we don't have one
            if not st.session_state["current_session"]:
                st.session_state["current_session"] = uuid.uuid4().hex[:16]

            # 3) Store transcription in MySQL
            store_transcription(st.session_state["current_session"], transcription_text)
            st.session_state["transcription"] = transcription_text

            # 4) Build vectorstore & retriever
            vs, rtr = build_vectorstore()
            st.session_state["vectorstore"] = vs
            st.session_state["retriever"] = rtr

            # 5) Reset memory & chain
            st.session_state["memory"] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            st.session_state["qa_chain"] = None
            initialize_qa_chain()

            # Mark transcription done
            st.session_state["audio_transcribed"] = True
    else:
        st.write("Please record audio using the button above.")
else:
    # If transcription is already done in this session, just show the transcription
    if st.session_state["transcription"]:
        st.subheader("Current Transcription")
        st.write(st.session_state["transcription"])

# --------------------------- #
#         DISPLAY CHAT        #
# --------------------------- #
# Show current chat messages
for msg in st.session_state["messages"]:
    avatar = USER_AVATAR if msg["role"] == "user" else BOT_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# --------------------------- #
#  MAIN CHAT INPUT & QA CHAIN #
# --------------------------- #
if st.session_state.get("qa_chain"):
    prompt = st.chat_input("Ask a new question here")
    if prompt:
        # 1) User message
        st.session_state["messages"].append({"role": "user", "content": prompt})
        store_chat_message(st.session_state["current_session"], "user", prompt)
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)

        # 2) Add user message to vectorstore
        add_document_to_vectorstore(prompt, source="chat", role="user")

        # 3) Get LLM response from chain
        response_dict = st.session_state["qa_chain"]({"question": prompt})
        answer = response_dict["answer"]

        # 4) Assistant message
        st.session_state["messages"].append({"role": "assistant", "content": answer})
        store_chat_message(st.session_state["current_session"], "assistant", answer)
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            st.markdown(answer)

        # 5) Add assistant response to vectorstore
        add_document_to_vectorstore(answer, source="chat", role="assistant")

        # 6) Update retriever if needed
        st.session_state["retriever"] = st.session_state["vectorstore"].as_retriever(
            search_kwargs={"filter": {"session_id": st.session_state["current_session"]}}
        )
