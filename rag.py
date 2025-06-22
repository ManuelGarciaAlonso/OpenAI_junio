# rag.py (Versión Final, Completa y Corregida)

import os
import gc
import time
import traceback
import re
import chromadb
from chromadb.config import Settings

from langchain_openai import ChatOpenAI, OpenAIEmbeddings, OpenAI  # Para completions API
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

class PDFChatbot:
    def __init__(self,
                 api_key: str,
                 chat_models=["o1-preview", "o1-mini", "gpt-4o", "gpt-4-turbo"],
                 embedding_model="text-embedding-3-small",
                 collection_name="langchain_fs_collection_openai"):
        
        print(f"🚀 Inicializando PDFChatbot con OpenAI...")
        if not api_key:
            raise ValueError("Se requiere una API Key de OpenAI.")

        self.api_key = api_key
        self.embedding_model_name = embedding_model
        
        self.embedding_service = OpenAIEmbeddings(model=self.embedding_model_name, openai_api_key=self.api_key)
        self.llm_instances = self._initialize_llms(chat_models)
        self.default_llm_name = chat_models[0]
        self.language_model = self.llm_instances.get(self.default_llm_name)
        
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        self.prompt_template = PromptTemplate.from_template(
            """<s>[INST] Eres un asistente docente experto en Fundamentos del Software. Tu tarea es responder preguntas sobre ejercicios y problemas de los diferentes temas.

Contexto:
{context}

Pregunta: {question}

Instrucciones de comportamiento:
1. Si te preguntan por ejercicios de un tema específico, enumera los ejercicios disponibles en ese tema.
2. Si te piden explicar un ejercicio, proporciona orientación como un profesor lo haría:
   - Explica los conceptos involucrados
   - Da pistas o sugerencias
   - NO proporciones la solución completa directamente
   - Guía al estudiante hacia la solución con preguntas reflexivas
3. Si te piden la solución directa, responde: "Como profesor, prefiero guiarte hacia la solución. ¿Qué parte específica del problema te está causando dificultades?"
4. Usa únicamente la información del contexto proporcionado.
5. Si la información no está en el contexto, di "No encontré información sobre eso en los materiales disponibles."

Recuerda, tu objetivo es enseñar, no solo proporcionar respuestas. [/INST] Respuesta:</s>"""
        )
        self.persist_directory = os.path.abspath("./chroma_db_openai")
        self.collection_name = collection_name
        
        # --- Lógica de Cliente Unificado ---
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(allow_reset=True)
        )
        
        self.vector_store = None
        self.retriever = None
        self.loaded_sources = []
        self.topic_index = {}  # Inicializar la propiedad
        self._load_existing_db_minimal()

    
    def _initialize_llms(self, model_names):
        instances = {}
        for name in model_names:
            try:
                model_kwargs = {}
                
                # Configuración especial para modelos de razonamiento
                if name in ["o1-preview", "o1-mini"]:
                    # Los modelos o1 no usan temperature
                    llm = ChatOpenAI(
                        model=name,
                        openai_api_key=self.api_key,
                        model_kwargs=model_kwargs
                    )
                elif name == "gpt-4o":
                    # Optimizado para razonamiento
                    llm = ChatOpenAI(
                        model=name,
                        openai_api_key=self.api_key,
                        temperature=0.1,  # Baja para más precisión
                        top_p=0.1,  # Más determinístico
                        model_kwargs=model_kwargs
                    )
                else:
                    llm = ChatOpenAI(
                        model=name,
                        openai_api_key=self.api_key,
                        temperature=0.1,
                        model_kwargs=model_kwargs
                    )

                instances[name] = llm
                print(f"✅ LLM de OpenAI '{name}' listo.")
            except Exception as e:
                print(f"❌ Error inicializando LLM '{name}': {e}.")
        return instances


    def clear_vectorstore(self):
        print("🔄 Reseteando la base de datos...")
        try:
            self.client.reset() 
            self.vector_store = Chroma(client=self.client, collection_name=self.collection_name, embedding_function=self.embedding_service)
            print("✅ Base de datos reseteada.")
            return True
        except Exception as e:
            print(f"❌ Falló el reseteo de la base de datos: {e}")
            return False

        
    def load_and_index_preprocessed(self, source_directory: str):
        """Carga, normaliza metadatos, procesa e indexa documentos."""
        print(f"📂 Cargando e indexando desde: {source_directory}")
        try:
            loader = DirectoryLoader(source_directory, glob="*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'}, show_progress=True)
            initial_docs = loader.load()
            if not initial_docs:
                print(f"⚠️ No se encontraron archivos en '{source_directory}'.")
                return []

            print(f"📄 {len(initial_docs)} archivos cargados.")

            # --- LA CORRECCIÓN CLAVE ESTÁ AQUÍ ---
            # Normalizamos los metadatos ANTES de procesar los documentos
            for doc in initial_docs:
                if 'source' in doc.metadata:
                    source_filename = os.path.basename(doc.metadata['source'])
                    # Extraer tema con regex
                    tema_match = re.search(r'ProblemasTema(\d+)(?:-(\d+))?', source_filename)
                    if tema_match:
                        tema_principal = tema_match.group(1)
                        tema_secundario = tema_match.group(2) or ""
                        tema_completo = f"{tema_principal}" if not tema_secundario else f"{tema_principal}-{tema_secundario}"
                        doc.metadata['tema'] = tema_completo
                    doc.metadata['source'] = source_filename
            # ------------------------------------

            print("🔪 Aplicando chunking semántico por ejercicios...")
            exercise_marker_regex = re.compile(r"^(### 🔖 Ejercicio (\d+)\.?\s*\n)", re.MULTILINE)
            all_chunks = []

            for doc in initial_docs:
                content = doc.page_content
                markers = list(exercise_marker_regex.finditer(content))

                # Procesar documentos con marcadores de ejercicio
                if markers:
                    # El contenido antes del primer marcador se trata por separado
                    if markers[0].start() > 0:
                        prefix_content = content[:markers[0].start()].strip()
                        if prefix_content:
                            all_chunks.extend(self.text_splitter.split_documents([prefix_content], metadatas=[doc.metadata]))

                    # Procesar cada ejercicio
                    for i, match in enumerate(markers):
                        exercise_number = match.group(2)
                        chunk_start = match.end()
                        chunk_end = markers[i+1].start() if i + 1 < len(markers) else len(content)
                        exercise_content = content[chunk_start:chunk_end].strip()

                        if exercise_content:
                            new_metadata = doc.metadata.copy()
                            new_metadata["exercise_number"] = exercise_number
                            split_chunks = self.text_splitter.create_documents([exercise_content], metadatas=[new_metadata])
                            all_chunks.extend(split_chunks)
                else:
                    # Procesar documentos sin marcadores de ejercicio
                    all_chunks.extend(self.text_splitter.split_documents([doc.page_content], metadatas=[doc.metadata]))

            if not all_chunks:
                print("⚠️ No se generaron chunks para indexar.")
                return []

            print(f"✅ Total {len(all_chunks)} chunks listos. Indexando en ChromaDB...")
            self.vector_store = Chroma.from_documents(
                documents=all_chunks,
                embedding=self.embedding_service,
                client=self.client,
                collection_name=self.collection_name
            )
            print("✅ Indexación completada.")
            self.loaded_sources = self._get_loaded_sources_from_db()
            self._setup_retriever()
            return self.loaded_sources
        except Exception as e:
            print(f"❌ Error CRÍTICO durante carga/indexación: {e}")
            traceback.print_exc()
            return []

    def answer_question(self, query, model_name=None):
        """
        Responde a una pregunta usando el sistema RAG.
        
        Args:
            query (str): La pregunta del usuario
            model_name (str, opcional): El nombre del modelo a utilizar. Si no se proporciona,
                                       se utilizará el modelo por defecto.
        
        Returns:
            str: La respuesta generada
        """
        # Si se especifica un modelo y existe en nuestros modelos disponibles, úsalo
        if model_name and model_name in self.llm_instances:
            llm_to_use = self.llm_instances[model_name]
        else:
            # Usa el primer modelo disponible como fallback
            model_name = list(self.llm_instances.keys())[0]
            llm_to_use = self.llm_instances[model_name]
            print(f"⚠️ Modelo solicitado no disponible, usando {model_name} como alternativa")
        
        # Obtener documentos relevantes
        retrieved_docs = self.retriever.invoke(query)
        
        # Formatear contexto
        context = self._format_context(retrieved_docs)
        
        # Construir la cadena de prompt
        chain = (
            {"context": lambda x: context, "question": lambda x: x}
            | self.prompt_template
            | llm_to_use
            | StrOutputParser()
        )
        
        # Generar respuesta
        response = chain.invoke(query)
        
        return response, retrieved_docs  # Devuelve tanto la respuesta como los documentos recuperados

    def _format_context(self, retrieved_docs):
        if not retrieved_docs: return "No se encontraron documentos relevantes."
        context_parts = []
        for i, doc in enumerate(retrieved_docs):
            source = os.path.basename(doc.metadata.get('source', 'N/A'))
            tema = doc.metadata.get('tema', 'No especificado')
            exercise = doc.metadata.get('exercise_number', None)
            header = f"--- Contexto {i+1} (Tema: {tema}, Ejercicio: {exercise}, Fuente: {source}) ---"
            context_parts.append(f"{header}\n{doc.page_content}")
        return "\n\n".join(context_parts)

    def _load_existing_db_minimal(self):
        try:
            self.vector_store = Chroma(client=self.client, collection_name=self.collection_name, embedding_function=self.embedding_service)
            if self.vector_store._collection.count() > 0:
                self.loaded_sources = self._get_loaded_sources_from_db()
                self._setup_retriever()
                print("✅ Sistema RAG listo con datos existentes.")
            else:
                 print("📂 Base de datos encontrada, pero está vacía.")
        except Exception as e:
            print(f"📂 No se encontró base de datos o hubo un error al cargarla: {e}. Se creará al indexar.")
            
    def _get_loaded_sources_from_db(self):
        metadatas = self.vector_store.get(include=["metadatas"]).get("metadatas", [])
        return sorted(list(set(os.path.basename(m['source']) for m in metadatas if 'source' in m)))

    
    def _setup_retriever(self, use_self_query=True):
        if not self.vector_store:
            print("⚠️ No se puede configurar el retriever, falta el vector_store.")
            return

        if use_self_query:
            try:
                print("🧠 Configurando retriever inteligente (SelfQuery) con instrucciones mejoradas...")

                # --- INSTRUCCIONES MEJORADAS Y EXPLÍCITAS ---
                source_description = (
                    "El nombre exacto del archivo fuente del que se extrajo el texto. "
                    "Los nombres de archivo siguen el patrón 'Ejercicios_ProblemasTemaX-Y.txt'. "
                    "Ejemplos: si el usuario pregunta por 'tema 1' o 'problemas del tema 1', el valor debe ser 'Ejercicios_ProblemasTema1.txt'. "
                    "Si pregunta por 'problemas del tema 2-2', el valor debe ser 'Ejercicios_ProblemasTema2-2.txt'."
                )

                metadata_field_info = [
                    AttributeInfo(
                        name="tema", 
                        description="El número del tema al que pertenece el ejercicio. Puede ser un número simple como '1', '2', '3' o compuesto como '2-2', '3-1'. Si la consulta menciona 'tema 1' buscar '1', si menciona 'tema 2-2' buscar '2-2'.", 
                        type="string"
                    ),
                    AttributeInfo(
                        name="source", 
                        description="El nombre del archivo fuente. Si se menciona un tema específico, buscar en archivos que contengan ese número de tema.", 
                        type="string"
                    ),
                    AttributeInfo(
                        name="exercise_number", 
                        description="El número del ejercicio específico, como '1', '2', etc. Si la consulta menciona 'ejercicio 5' o 'problema 5', buscar este valor.", 
                        type="string"
                    ),
                ]

                doc_content_desc = "Fragmentos de texto de ejercicios del curso Fundamentos del Software."

                self.retriever = SelfQueryRetriever.from_llm(
                    llm=self.language_model,
                    vectorstore=self.vector_store,
                    document_contents=doc_content_desc,
                    metadata_field_info=metadata_field_info,
                    verbose=True
                )
                print("✅ SelfQueryRetriever configurado exitosamente.")
            except Exception as e:
                print(f"⚠️ Falló la configuración de SelfQuery, usando retriever básico como fallback: {e}")
                self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        else:
            print("🔄 Configurando retriever básico (VectorSearch)...")
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})

    def _preprocess_query(self, query):
        """Analiza la consulta para identificar temas y ejercicios específicos."""
        tema_match = re.search(r'tema\s+(\d+)(?:-(\d+))?', query.lower())
        ejercicio_match = re.search(r'ejercicio\s+(\d+)|problema\s+(\d+)', query.lower())
        
        filter_parts = []
        if tema_match:
            tema_principal = tema_match.group(1)
            tema_secundario = tema_match.group(2)
            tema_str = f"{tema_principal}" if not tema_secundario else f"{tema_principal}-{tema_secundario}"
            filter_parts.append(f"tema == '{tema_str}'")
        
        if ejercicio_match:
            ejercicio = ejercicio_match.group(1) or ejercicio_match.group(2)
            filter_parts.append(f"exercise_number == '{ejercicio}'")
        
        return query, " AND ".join(filter_parts) if filter_parts else None

    def build_topic_index(self):
        """Crea un índice de temas y ejercicios disponibles para referencia rápida."""
        self.topic_index = {}
        if not self.vector_store:
            print("⚠️ No se puede construir índice de temas: Vector store no inicializado")
            return self.topic_index
            
        try:
            metadatas = self.vector_store.get(include=["metadatas"]).get("metadatas", [])
            for meta in metadatas:
                if 'tema' in meta and 'exercise_number' in meta:
                    tema = meta['tema']
                    ejercicio = meta['exercise_number']
                    if tema not in self.topic_index:
                        self.topic_index[tema] = set()
                    self.topic_index[tema].add(ejercicio)
            print(f"✅ Índice de temas construido: {len(self.topic_index)} temas encontrados")
        except Exception as e:
            print(f"❌ Error al construir índice de temas: {e}")
        
        return self.topic_index

    def list_available_topics(self):
        """Devuelve una lista formateada de temas y ejercicios disponibles."""
        if not hasattr(self, 'topic_index'):
            self.build_topic_index()
        
        result = "Temas y ejercicios disponibles:\n\n"
        for tema in sorted(self.topic_index.keys()):
            ejercicios = sorted(list(self.topic_index[tema]), key=lambda x: int(x))
            result += f"Tema {tema}: Ejercicios {', '.join(ejercicios)}\n"
        return result

    def load_existing_data(self, directory_path):
        """Carga datos existentes desde un directorio previamente procesado."""
        print(f"📂 Cargando datos existentes desde: {directory_path}")
        return self.load_and_index_preprocessed(directory_path)