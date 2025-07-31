import os
import time
import shelve
import tempfile
import requests
import numpy as np
import soundfile as sf
import torch
import json
from scipy import signal
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import streamlit as st
from audio_recorder_streamlit import audio_recorder
import google.generativeai as genai
from streamlit.components.v1 import html

USER_AVATAR = "👤"
BOT_AVATAR = "❤️‍🩹"


ASSEMBLYAI_API_KEY="e6a046986c1e4f57b16da9d3f7a2fb1d"

GEMINI_API_KEY = "AIzaSyC57_75ViKwP7p9fnmlrnQCRn3maBr0L2M"

# API Endpoints
UPLOAD_ENDPOINT = "https://api.assemblyai.com/v2/upload"
TRANSCRIBE_ENDPOINT = "https://api.assemblyai.com/v2/transcript"

# Emotion Model
EMOTION_MODEL_NAME = "Dpngtm/wav2vec2-emotion-recognition"
TARGET_SAMPLE_RATE = 16000
EMOTION_LABELS = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
POSITIVE_EMOTIONS = ['happy', 'neutral', 'calm', 'surprised']

# Gemini Model
GEMINI_MODEL = "gemini-1.5-flash"
DISCLAIMER_TEXT = "Please note: I am an AI companion, not a substitute for professional medical or mental health advice. If you are in crisis, please contact a suicide prevention hotline or seek professional help immediately."

SYSTEM_INSTRUCTION = """
You are HealthPal, a compassionate, emotionally intelligent AI companion **specifically designed to provide support and information related to mental and physical well-being**. Your role is to support users like a deeply caring, non-judgmental friend, **focusing exclusively on health-related queries.**

Your tone must adapt to the user's emotional state based on their message:
- If the user seems sad, anxious, stressed, or is sharing a problem, respond with warmth, empathy, and care. Offer comfort and gentle reflections.
- If the user seems happy, positive, or cheerful, share their joy! Respond with enthusiasm, encouragement, and a light, friendly tone.
- If you receive context from a voice analysis, use that as a strong hint for the user's emotional state.
- Always maintain a supportive and non-judgmental persona. Your primary goal is to make the user feel heard, understood, and supported, whatever their mood.

**IMPORTANT:** If a user's query is outside the scope of mental or physical health, gently guide them back to topics relevant to your purpose. For example, you might say, "I'm here to talk about your well-being. How can I support you with your mental or physical health today?" Do not engage in conversations unrelated to health.

🧾 Format:
You must respond ONLY with a single valid JSON object like this, with no extra text or markdown:
{
  "answer": "Your adaptive, emotionally aware plain-text response goes here.",
  "suggestions": ["User's potential question 1", "User's potential question 2", "User's potential question 3"]
}

- The items in the "suggestions" list MUST be questions a user would ask you, the AI, to get further help or guidance.
- The questions should be written from the user's perspective. For example, for a user who said "I'm depressed," the suggestions should be questions THEY might ask YOU, such as:
    - "What are some things I can do to feel better?"
    - "Can you tell me more about managing these feelings?"
    - "How can I get professional help?"
- DO NOT generate questions for the AI to ask the user. The suggestions are for the user to click and continue the conversation.
- If no logical follow-up questions from the user exist, return an empty list: "suggestions": []
- Use only plain, emotionally resonant language. Avoid markdown or formatting symbols.
"""

# =============================
# 🎨 --- UI / STYLING ---
# =============================
def apply_custom_css():
    st.markdown("""
        <style>
            /* Chat bubbles */
            .st-emotion-cache-1c7y2kd {
                background-color: #FFFFFF;
                border-radius: 20px;
                padding: 1rem 1.25rem;
            }
            [data-testid="stChatMessage"]:has([data-testid="stAvatarIcon-user"]) .st-emotion-cache-1c7y2kd {
                background-color: #DCF8C6;
            }

            /* Suggestion buttons (red) */
            .stButton > button {
                border-radius: 20px;
                border: 1px solid #D9534F; color: #D9534F;
                background-color: transparent;
            }
            .stButton > button:hover {
                border-color: #D9534F; color: #FFFFFF; background-color: #D9534F;
            }
            
            /* Clear Chat button (red) */
            [data-testid="stButton-clear_chat_button"] > button {
                border-color: #D9534F; color: #FFFFFF; background-color: #D9534F; width: 100%;
            }
            [data-testid="stButton-clear_chat_button"] > button:hover {
                background-color: #C9302C; border-color: #C9302C;
            }

            /* Disclaimer text */
            [data-testid="stChatMessage"] p:contains("Please note:") {
                font-size: 0.8rem; font-style: italic; color: #666666; text-align: center;
                padding: 0.5rem; background-color: #FAFAFA; border-radius: 10px;
            }
        </style>
    """, unsafe_allow_html=True)

# =============================
# 🧠 --- STATE & PERSISTENCE ---
# =============================
def load_chat_history():
    with shelve.open("chat_history_healthpal") as db:
        return db.get("messages", [])

def save_chat_history(messages):
    with shelve.open("chat_history_healthpal") as db:
        db["messages"] = messages

if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()
    if not st.session_state.messages:
        st.session_state.messages.append({"role": "assistant", "content": DISCLAIMER_TEXT, "suggestions": []})
        save_chat_history(st.session_state.messages)

if "processing" not in st.session_state: st.session_state.processing = False
if "last_audio_hash" not in st.session_state: st.session_state.last_audio_hash = None

# =======================================================================
# 📦 --- HELPER FUNCTIONS (FULLY IMPLEMENTED) ---
# =======================================================================

def auto_scroll():
    js = """
    <script>
        function scrollToBottom() {
            const chatContainer = parent.document.querySelector('.st-emotion-cache-1jicfl2');
            if (chatContainer) {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        }
        window.addEventListener('load', scrollToBottom);
        const observer = new MutationObserver(scrollToBottom);
        const config = { childList: true, subtree: true };
        const targetNode = parent.document.querySelector('[data-testid="stVerticalBlock"]');
        if (targetNode) { observer.observe(targetNode, config); }
    </script>
    """
    html(js, height=0)

def save_audio_file(audio_bytes, suffix=".wav"):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            return tmp.name
    except Exception as e:
        st.error(f"Error saving audio: {e}"); return None

def transcribe_with_assemblyai(audio_file_path):
    try:
        headers = {"authorization": ASSEMBLYAI_API_KEY}
        with open(audio_file_path, "rb") as f:
            upload_response = requests.post(UPLOAD_ENDPOINT, headers=headers, data=f)
        upload_response.raise_for_status()
        upload_url = upload_response.json()["upload_url"]

        json_data = {"audio_url": upload_url}
        transcribe_response = requests.post(TRANSCRIBE_ENDPOINT, headers=headers, json=json_data)
        transcribe_response.raise_for_status()
        transcript_id = transcribe_response.json()["id"]
        
        polling_endpoint = f"{TRANSCRIBE_ENDPOINT}/{transcript_id}"
        while True:
            result = requests.get(polling_endpoint, headers=headers).json()
            if result['status'] == 'completed': return result['text']
            if result['status'] == 'error': raise RuntimeError(f"Transcription failed: {result['error']}")
            time.sleep(2)
    except Exception as e:
        st.error(f"Transcription error: {e}"); return None

@st.cache_resource(show_spinner="Loading emotion detection model...")
def load_emotion_detector():
    try:
        detector = Wav2Vec2ForSequenceClassification.from_pretrained(EMOTION_MODEL_NAME)
        processor = Wav2Vec2Processor.from_pretrained(EMOTION_MODEL_NAME)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        detector.to(device)
        return detector, processor, device
    except Exception as e:
        st.error(f"Emotion model load error: {e}"); return None, None, None

def analyze_audio_input(audio_file_path):
    detector, processor, device = load_emotion_detector()
    if not detector: return None, None, None
    try:
        audio_array, sr = sf.read(audio_file_path)
        if len(audio_array.shape) > 1: audio_array = np.mean(audio_array, axis=1)
        if sr != TARGET_SAMPLE_RATE:
            num_samples = round(len(audio_array) * TARGET_SAMPLE_RATE / sr)
            audio_array = signal.resample(audio_array, num_samples)
        
        inputs = processor(audio_array, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad(): logits = detector(**inputs).logits
        
        probs = torch.nn.functional.softmax(logits, dim=1)[0]
        emotion = EMOTION_LABELS[torch.argmax(probs).item()]
        scores = {EMOTION_LABELS[i]: round(p.item() * 100, 2) for i, p in enumerate(probs)}
        sentiment = 'positive' if emotion in POSITIVE_EMOTIONS else 'negative'
        return emotion, sentiment, scores
    except Exception as e:
        st.error(f"Audio analysis error: {e}"); return None, None, None

@st.cache_resource(show_spinner=False)
def build_gemini_client():
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel(GEMINI_MODEL)

def generate_response(prompt, emotion_context=None):
    try:
        model = build_gemini_client()
        context_prompt = f"{emotion_context}\n\n" if emotion_context else ""
        full_prompt = f"{SYSTEM_INSTRUCTION}\n\n{context_prompt}User query: {prompt}"
        
        resp = model.generate_content(full_prompt)
        
        if hasattr(resp, "text") and resp.text:
            cleaned_text = resp.text.strip().replace("```json", "").replace("```", "")
            try:
                data = json.loads(cleaned_text)
                return data.get("answer", "I'm sorry, I couldn't formulate a response."), data.get("suggestions", [])
            except json.JSONDecodeError:
                return cleaned_text, [] 
        return "(No response from model)", []
    except Exception as e:
        st.error(f"Gemini API error: {e}"); return f"Sorry, an error occurred: {e}", []

def add_message(role, content, **kwargs):
    st.session_state.messages.append({"role": role, "content": content, **kwargs})
    save_chat_history(st.session_state.messages)

def render_chat_messages():
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"], avatar=(USER_AVATAR if message["role"] == "user" else BOT_AVATAR)):
            st.markdown(message["content"])
            if message["role"] == "user" and message.get("emotion"):
                with st.expander("🎙️ Voice Analysis"):
                    st.json({
                        "Emotion": message["emotion"],
                        "Sentiment": message["sentiment"],
                        "Scores": message["emotion_scores"]
                    })
            is_last = (i == len(st.session_state.messages) - 1)
            if message["role"] == "assistant" and message.get("suggestions") and is_last:
                cols = st.columns(len(message["suggestions"]))
                for j, suggestion in enumerate(message["suggestions"]):
                    if cols[j].button(suggestion, key=f"s_{i}_{j}"):
                        handle_input(suggestion)
                        st.rerun()

def handle_input(prompt, is_audio=False, audio_bytes=None):
    if not prompt or st.session_state.processing: return
    st.session_state.processing = True
    
    if is_audio:
        with st.spinner("Analyzing your voice..."):
            audio_file = save_audio_file(audio_bytes, ".wav")
            if audio_file:
                transcript = transcribe_with_assemblyai(audio_file)
                if transcript:
                    emotion, sentiment, scores = analyze_audio_input(audio_file)
                    add_message("user", transcript, emotion=emotion, sentiment=sentiment, emotion_scores=scores)
                    emotion_context = f"Context from voice analysis: The user's tone sounds {emotion}."
                    answer, suggestions = generate_response(transcript, emotion_context)
                    add_message("assistant", answer, suggestions=suggestions)
                if os.path.exists(audio_file): os.unlink(audio_file)
    else:
        add_message("user", prompt)
        with st.spinner("Thinking..."):
            answer, suggestions = generate_response(prompt)
        add_message("assistant", answer, suggestions=suggestions)

    st.session_state.processing = False

# =============================
# หลัก --- MAIN APP ---
# =============================
def main():
    st.set_page_config(page_title="Chat Bot", page_icon="🧠", layout="centered")
    apply_custom_css()

    st.markdown("<h1 style='text-align: center; font-weight: 800;'>🧠 Mental Health Chat Bot</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; margin-top: -10px; color: #666;'>Your compassionate AI therapy companion</p>", unsafe_allow_html=True)
    
    with st.sidebar:
        st.subheader("🎤 Voice Input")
        audio_bytes = audio_recorder(text="", icon_size="1.5x", key="audio_recorder")
        
        st.subheader("🧹 Chat Controls")
        if st.button("Clear & Restart Chat", key="clear_chat_button"):
            st.session_state.messages = [{"role": "assistant", "content": DISCLAIMER_TEXT, "suggestions": []}]
            save_chat_history(st.session_state.messages)
            st.rerun()

    chat_container = st.container(height=500, border=False)
    with chat_container:
        render_chat_messages()
    
    auto_scroll()

    # --- INPUT HANDLING ---
    if audio_bytes:
        audio_hash = hash(audio_bytes)
        if st.session_state.last_audio_hash != audio_hash:
            st.session_state.last_audio_hash = audio_hash
            handle_input("Audio input", is_audio=True, audio_bytes=audio_bytes)
            st.rerun()

    if prompt := st.chat_input("How are you feeling today?"):
        handle_input(prompt)
        st.rerun()

if __name__ == "__main__":
    main()