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
# Modelos disponibles en la suscripción Tier 1 de OpenAI
AVAILABLE_MODELS = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4"]
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
    
    # Selector de temperatura
    temperature = st.slider(
        "Temperatura (creatividad):",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        help="Valores más bajos = respuestas más precisas. Valores más altos = respuestas más creativas."
    )
    
    # Botón para inicializar/reiniciar chatbot
    if st.button("🔄 Inicializar/Reiniciar Chatbot"):
        with st.spinner("Inicializando sistema..."):
            try:
                api_key = os.getenv("OPENAI_API_KEY")
                st.session_state.pdf_chatbot = PDFChatbot(
                    api_key=api_key,
                    chat_models=AVAILABLE_MODELS,
                    embedding_model=DEFAULT_EMBEDDING_MODEL,
                    temperature=temperature  # Pasar la temperatura seleccionada
                )
                
                # Cargar datos si existen
                if os.path.exists(PREPROCESSED_DATA_DIR):
                    st.session_state.pdf_chatbot.load_existing_data(PREPROCESSED_DATA_DIR)
                
                st.success("¡Chatbot inicializado correctamente!")
            except Exception as e:
                st.error(f"Error al inicializar: {e}")
    
    # Botón para limpiar conversación
    if st.button("🧹 Limpiar conversación"):
        st.session_state.messages = []
        st.rerun()
    
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
                # Pasar mensajes previos para contexto
                previous_messages = st.session_state.messages[:-1] if len(st.session_state.messages) > 1 else []
                
                response_data = st.session_state.pdf_chatbot.answer_question(
                    user_input, 
                    model_name=st.session_state.selected_model,
                    previous_messages=previous_messages
                )
                
                # Extraer respuesta y documentos recuperados
                if isinstance(response_data, tuple) and len(response_data) >= 1:
                    response_text = response_data[0]
                    retrieved_docs = response_data[1] if len(response_data) > 1 else []
                else:
                    response_text = response_data
                    retrieved_docs = []
                
                # Guardar documentos recuperados para referencia
                st.session_state.last_retrieved_docs = retrieved_docs
                
                # Añadir respuesta al historial
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                
                # Mostrar las fuentes consultadas
                if retrieved_docs:
                    with st.expander("🔍 Ver fuentes consultadas"):
                        for i, doc in enumerate(retrieved_docs):
                            tema = doc.metadata.get('tema', 'No especificado')
                            ejercicio = doc.metadata.get('exercise_number', 'No especificado')
                            source = doc.metadata.get('source', 'Desconocido')
                            
                            st.markdown(f"**Fuente {i+1}:** Tema {tema}, Ejercicio {ejercicio}, Archivo: {source}")
                            
                            # Mostrar fragmento del contenido para verificación
                            with st.expander(f"Ver fragmento del contenido"):
                                st.text(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
                
            except Exception as e:
                error_msg = f"Error al procesar tu consulta: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": f"⚠️ {error_msg}"})
    
    # Recargar para mostrar la nueva respuesta
    st.rerun()

# Después de mostrar los mensajes existentes
if "last_retrieved_docs" in st.session_state and st.session_state.last_retrieved_docs:
    with st.expander("🔍 Ver fuentes de la última consulta"):
        for i, doc in enumerate(st.session_state.last_retrieved_docs):
            tema = doc.metadata.get('tema', 'No especificado')
            ejercicio = doc.metadata.get('exercise_number', 'No especificado')
            st.markdown(f"**Fuente {i+1}:** Tema {tema}, Ejercicio {ejercicio}")