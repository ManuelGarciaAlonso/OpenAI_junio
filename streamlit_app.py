# streamlit_app.py (Versión Final, Completa y Corregida)

import os
import streamlit as st
from rag import PDFChatbot
from streamlit_chat import message
from dotenv import load_dotenv
import time

# --- Carga de variables de entorno ---
load_dotenv()

# --- Constantes ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSED_DATA_DIR = os.path.join(APP_DIR, "Processed_Texts", "preprocessed_markdown")
AVAILABLE_MODELS = ["o1-preview", "o1-mini", "gpt-4o", "gpt-4-turbo"]
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

st.set_page_config(page_title="ChatBot FS (OpenAI)", layout="wide")

# --- Inicialización básica de la sesión ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_chatbot" not in st.session_state:
    st.session_state.pdf_chatbot = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gpt-4o"
if "last_retrieved_docs" not in st.session_state:
    st.session_state.last_retrieved_docs = []

# --- Configuración de la interfaz ---
st.title("🤖 Asistente de Fundamentos del Software")

# --- Sidebar con controles ---
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # Selector de modelo
    selected_model = st.selectbox(
        "Selecciona el modelo:",
        AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(st.session_state.selected_model)
    )
    st.session_state.selected_model = selected_model
    
    # Botón para inicializar/reiniciar chatbot
    if st.button("🔄 Inicializar/Reiniciar Chatbot"):
        with st.spinner("Inicializando sistema..."):
            try:
                api_key = os.getenv("OPENAI_API_KEY")
                st.session_state.pdf_chatbot = PDFChatbot(
                    api_key=api_key,
                    chat_models=AVAILABLE_MODELS,
                    embedding_model=DEFAULT_EMBEDDING_MODEL
                )
                
                # Cargar datos si existen
                if os.path.exists(PREPROCESSED_DATA_DIR):
                    st.session_state.pdf_chatbot.load_existing_data(PREPROCESSED_DATA_DIR)
                
                st.success("¡Chatbot inicializado correctamente!")
            except Exception as e:
                st.error(f"Error al inicializar: {e}")
    
    # Ver temas disponibles
    if st.button("📚 Ver temas disponibles"):
        if st.session_state.pdf_chatbot:
            with st.spinner("Generando índice de temas..."):
                try:
                    # Usar el método real que existe en PDFChatbot
                    topics_info = st.session_state.pdf_chatbot.list_available_topics()
                    st.info(topics_info)
                except Exception as e:
                    st.error(f"Error al mostrar temas: {e}")

# --- Área de chat ---
# Mostrar mensajes existentes
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        message(msg["content"], is_user=True, key=f"user_msg_{i}")
    else:
        message(msg["content"], is_user=False, key=f"ai_msg_{i}")

# Input del usuario
user_input = st.chat_input("Escribe tu pregunta aquí...")

# Procesar input cuando se envía
if user_input:
    # Añadir mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Verificar si el chatbot está inicializado
    if not st.session_state.pdf_chatbot:
        st.error("Por favor, inicializa el chatbot primero usando el botón en la barra lateral.")
        st.session_state.messages.append({"role": "assistant", "content": "⚠️ Por favor, inicializa el chatbot primero usando el botón en la barra lateral."})
    else:
        # Procesar la consulta
        with st.spinner("Procesando tu consulta..."):
            try:
                # La función devuelve (response, retrieved_docs)
                response_data = st.session_state.pdf_chatbot.answer_question(
                    user_input, 
                    model_name=st.session_state.selected_model
                )
                
                # Extraer respuesta y documentos de manera más robusta
                if isinstance(response_data, tuple) and len(response_data) >= 1:
                    response_text = response_data[0]
                    retrieved_docs = response_data[1] if len(response_data) > 1 else []
                else:
                    response_text = response_data  # En caso de que solo devuelva texto
                    retrieved_docs = []
                
                # Guardar documentos recuperados para posible uso futuro
                st.session_state.last_retrieved_docs = retrieved_docs
                
                # Añadir respuesta al historial
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            except Exception as e:
                error_msg = f"Error al procesar tu consulta: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": f"⚠️ {error_msg}"})
    
    # Recargar para mostrar la nueva respuesta
    st.rerun()