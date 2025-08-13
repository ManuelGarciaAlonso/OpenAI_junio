# Asistente Docente de Fundamentos del Software

Este repositorio contiene un asistente docente basado en IA, diseñado para apoyar el aprendizaje en la asignatura "Fundamentos del Software". El asistente está implementado como una aplicación web con Streamlit y utiliza RAG (Retrieval Augmented Generation) para responder consultas sobre ejercicios y problemas de la asignatura.

## 📋 Características

- **Enfoque pedagógico**: Guía a los estudiantes a través del razonamiento, en lugar de proporcionar respuestas directas
- **Organización por temas**: Acceso a ejercicios organizados por temas y subtemas
- **Recuperación contextual**: Mantiene el contexto de la conversación para responder coherentemente
- **Adaptabilidad**: Permite seleccionar diferentes modelos de OpenAI

## 🔧 Requisitos previos

- Python 3.9+
- Una cuenta en [OpenAI](https://platform.openai.com/) con crédito disponible
- API Key de OpenAI con acceso a modelos GPT (gpt-4o, gpt-4-turbo, etc.)

## 🚀 Instalación

1. **Clonar el repositorio**:
   ```bash
   git clone https://github.com/ManuelGarciaAlonso/OpenAI_junio.git
   cd TFM-OpenAI
   ```

2. **Crear y activar un entorno virtual**:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configurar la API Key de OpenAI**:
   
   Crea un archivo .env en la raíz del proyecto con el siguiente contenido:
   ```
   OPENAI_API_KEY=tu_api_key_aquí
   ```

## 📂 Estructura del proyecto

```
.
├── streamlit_app.py      # Aplicación principal de Streamlit
├── rag.py                # Implementación del sistema RAG
├── Processed_Texts/      # Documentos procesados
│   └── preprocessed_markdown/  # Textos en formato markdown preprocesados
├── chroma_db_openai/     # Base de datos vectorial (generada automáticamente)
├── requirements.txt      # Dependencias del proyecto
└── .env                  # Archivo de variables de entorno (crear manualmente)
```

## 📝 Preparación de los documentos

1. **Estructura de los documentos**:
   - Los documentos deben estar en formato `.txt`
   - Nombrados como `Ejercicios_ProblemasTemaX-Y.txt` donde X es el tema principal e Y el subtema (opcional)
   - Cada ejercicio debe estar marcado con `### 🔖 Ejercicio N` donde N es el número del ejercicio

2. **Ubicación de los documentos**:
   - Coloca los documentos en la carpeta preprocessed_markdown

## 🖥️ Ejecución de la aplicación

1. **Iniciar la aplicación Streamlit**:
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Acceder a la aplicación**:
   - Abre tu navegador en `http://localhost:8501`
   - La primera vez, haz clic en "Inicializar/Reiniciar Chatbot" en el panel lateral

3. **Interactuar con el chatbot**:
   - Selecciona el modelo de OpenAI en el panel lateral
   - Ajusta la temperatura según prefieras más precisión o creatividad
   - Haz consultas sobre ejercicios específicos o temas

## 🔄 Funciones principales

- **Ver temas disponibles**: Muestra todos los temas y ejercicios indexados
- **Consultar ejercicios**: Pregunta por ejercicios específicos como "Dame el ejercicio 3 del tema 2"
- **Solicitar explicaciones**: El chatbot guía sin dar respuestas directas
- **Verificar respuestas**: Si propones una respuesta, el chatbot te orientará sobre si es correcta

## 💡 Ejemplos de uso

- "¿Qué ejercicios hay disponibles en el tema 2?"
- "Explícame el ejercicio 3 del tema 3-2"
- "Dame una pista para resolver el ejercicio 1 del tema 2-2"
- "La respuesta del ejercicio anterior es la opción B"
- "¿Por qué es incorrecta mi respuesta?"

## ⚙️ Personalización

- **Modelos disponibles**: Edita la lista `AVAILABLE_MODELS` en streamlit_app.py según tu suscripción
- **Temperatura**: Ajusta la creatividad del modelo con el control deslizante de temperatura
- **Prompts**: Modifica el sistema de prompts en rag.py para ajustar el comportamiento del asistente

## 📚 Notas adicionales

- La primera ejecución puede tardar más tiempo mientras se construye la base de datos vectorial
- Asegúrate de tener suficiente crédito en tu cuenta de OpenAI para usar los modelos seleccionados
- La aplicación guarda el historial de la conversación solo durante la sesión actual

## 🤝 Contribución

Este es un repositorio privado. Para contribuir, contacta directamente con el propietario.

---

Desarrollado como parte del Trabajo de Fin de Máster en Ingeniería Informática.