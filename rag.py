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
                 chat_models=["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4"],
                 embedding_model="text-embedding-3-small",
                 collection_name="langchain_fs_collection_openai",
                 temperature=0.1):  # Añadir parámetro de temperatura
        
        print(f"🚀 Inicializando PDFChatbot con OpenAI...")
        if not api_key:
            raise ValueError("Se requiere una API Key de OpenAI.")

        self.api_key = api_key
        self.embedding_model_name = embedding_model
        self.temperature = temperature  # Guardar la temperatura
        self.collection_name = collection_name
        
        self.embedding_service = OpenAIEmbeddings(model=self.embedding_model_name, openai_api_key=self.api_key)
        self.llm_instances = self._initialize_llms(chat_models)
        self.default_llm_name = chat_models[0]
        self.language_model = self.llm_instances.get(self.default_llm_name)
        
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        self.prompt_template = PromptTemplate.from_template(
            """<s>[INST] Eres un asistente docente experto en Fundamentos del Software. Tu tarea es responder consultas sobre ejercicios y problemas de los diferentes temas.

Contexto:
{context}

Consulta: {question}

Instrucciones de comportamiento:
1. Si te preguntan por una lista de ejercicios de un tema específico (ej: "¿Qué ejercicios hay en el tema 2?"):
   - Proporciona la lista completa de ejercicios disponibles para ese tema
   - Si el tema tiene subtemas, menciónalos también (ej: tema 2-1, tema 2-2)
2. Si te piden explicar un ejercicio, proporciona orientación como un profesor lo haría:
   - Explica los conceptos involucrados
   - Da pistas o sugerencias
   - NO proporciones la solución completa directamente
   - Guía al estudiante hacia la solución con preguntas reflexivas
3. Si el usuario indica que la respuesta es una opción específica (ej: "la respuesta es c"):
   - Verifica si esa opción es correcta según el contexto
   - Si es correcta, confirma y explica por qué es la respuesta adecuada
   - Si es incorrecta, NO des la respuesta correcta directamente. En su lugar, pregunta "¿Por qué piensas que es esa opción?" y da pistas sutiles sobre qué considerar
   - Si no puedes verificarla, pide aclaración
4. Usa únicamente la información del contexto proporcionado.
5. Si la información no está en el contexto, di "No encontré información sobre eso en los materiales disponibles."

Recuerda: Tu objetivo es que el estudiante APRENDA, no simplemente obtener respuestas correctas. Prioriza el proceso de aprendizaje por encima de todo.

Instrucciones adicionales:
1. Cuando te pregunten por un tema (ej: "tema 3"), considera tanto los ejercicios del tema base como sus subtemas (3-1, 3-2, etc).
2. Si el usuario menciona un tema general como "tema 1", incluye en tu respuesta información sobre los subtemas disponibles.
3. Cuando la consulta sea corta o ambigua (como "ok", "la respuesta es c"), relaciónala con la conversación anterior.
4. Responde de forma concisa usando exclusivamente la información del contexto.
5. Si la información no aparece en el contexto, indícalo claramente. [/INST] Respuesta:</s>"""
        )
        self.persist_directory = os.path.abspath("./chroma_db_openai")
        
        # --- Lógica de Cliente Unificado ---
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(allow_reset=True)
        )
        
        self.vector_store = None
        self.retriever = None
        self.loaded_sources = []
        self.topic_index = {}  # Inicializar la propiedad
        self.conversation_history = []  # Para mantener contexto entre consultas
        self.token_usage = {"total": 0, "by_model": {}}  # Inicializar seguimiento de uso de tokens
        self._load_existing_db_minimal()

    
    def _initialize_llms(self, model_names):
        instances = {}
        for name in model_names:
            try:
                model_kwargs = {}
                
                llm = ChatOpenAI(
                    model=name,
                    openai_api_key=self.api_key,
                    temperature=self.temperature,  # Usar la temperatura configurada
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
            # Cuando procesas los documentos, añade tanto el tema completo como el tema principal
            for doc in initial_docs:
                if 'source' in doc.metadata:
                    source_filename = os.path.basename(doc.metadata['source'])
                    # Extraer tema con regex
                    tema_match = re.search(r'ProblemasTema(\d+)(?:-(\d+))?', source_filename)
                    if tema_match:
                        tema_principal = tema_match.group(1)
                        tema_secundario = tema_match.group(2) or ""
                        tema_completo = f"{tema_principal}" if not tema_secundario else f"{tema_principal}-{tema_secundario}"
                        
                        # Guardar ambos valores como metadatos
                        doc.metadata['tema'] = tema_completo
                        doc.metadata['tema_principal'] = tema_principal
                    
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
                            all_chunks.extend(self.text_splitter.split_documents([Document(page_content=prefix_content, metadata=doc.metadata)]))

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
                    all_chunks.extend(self.text_splitter.split_documents([Document(page_content=doc.page_content, metadata=doc.metadata)]))

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

    def answer_question(self, query, model_name=None, use_memory=True, previous_messages=None):
        """
        Responde a una pregunta usando el sistema RAG.
        
        Args:
            query (str): La pregunta del usuario
            model_name (str, opcional): El nombre del modelo a utilizar
            use_memory (bool): Si se debe usar el historial de conversación
            previous_messages (list): Mensajes previos de la conversación
        """
        # Verificar si es una consulta especial de listar ejercicios
        if query.startswith("list_exercises:"):
            tema = query.split(":", 1)[1]
            return self.get_exercises_by_topic(tema), []
        
        # Si se especifica un modelo y existe en nuestros modelos disponibles, úsalo
        if model_name and model_name in self.llm_instances:
            llm_to_use = self.llm_instances[model_name]
        else:
            # Usa el primer modelo disponible como fallback
            model_name = list(self.llm_instances.keys())[0]
            llm_to_use = self.llm_instances[model_name]
            print(f"⚠️ Modelo solicitado no disponible, usando {model_name} como alternativa")
        
        # Preprocesar la consulta considerando el contexto conversacional
        processed_query, filter_str, conversation_context = self._preprocess_query_with_context(query, previous_messages)
        
        # Si no se encontraron documentos con el filtro exacto, intentar con una estrategia de respaldo
        if filter_str and "tema_principal" in filter_str:
            # Extraer el tema principal
            tema_principal_match = re.search(r"tema_principal == '(\d+)'", filter_str)
            if tema_principal_match:
                tema_principal = tema_principal_match.group(1)
                
                # Primero intentar con el filtro normal
                retrieved_docs = self.retriever.invoke(processed_query, filter=filter_str)
                
                # Si no hay resultados, intentar una búsqueda más general
                if not retrieved_docs:
                    print(f"⚠️ No se encontraron documentos para tema_principal={tema_principal}, buscando de forma alternativa...")
                    # Buscar en todos los documentos por similaridad semántica y filtrar después por código
                    all_docs = self.retriever.invoke(processed_query)
                    retrieved_docs = [doc for doc in all_docs if 
                                     doc.metadata.get('tema', '').startswith(tema_principal) or
                                     f"Tema{tema_principal}" in doc.metadata.get('source', '')]
        else:
            # Busqueda normal
            retrieved_docs = self.retriever.invoke(processed_query, filter=filter_str) if filter_str else self.retriever.invoke(processed_query)
        
        # Si no se encontraron documentos, realizar búsqueda alternativa
        if not retrieved_docs:
            print("⚠️ No se encontraron documentos con el filtro especificado, probando búsqueda alternativa...")
            
            # 1. Intentar relajar los filtros
            if filter_str:
                filter_parts = filter_str.split(" AND ")
                if len(filter_parts) > 1:
                    # Usar solo uno de los filtros (por ejemplo, solo el número de ejercicio o solo el tema)
                    for single_filter in filter_parts:
                        alt_docs = self.retriever.invoke(processed_query, filter=single_filter)
                        if alt_docs:
                            print(f"✅ Encontrados {len(alt_docs)} documentos usando filtro relajado: {single_filter}")
                            retrieved_docs = alt_docs
                            break
            
            # 2. Si aún no hay resultados, realizar una búsqueda sin filtros pero filtrando después
            if not retrieved_docs:
                all_docs = self.retriever.invoke(processed_query)
                
                # Si la consulta menciona una respuesta (como "la respuesta es c")
                answer_match = re.search(r'la respuesta es (\w)', query.lower())
                if answer_match and conversation_context:
                    # Filtrar documentos que tengan el mismo ejercicio que se discutía
                    exercise_match = re.search(r'ejercicio (\d+) del tema (\d+)', conversation_context.lower())
                    if exercise_match:
                        exercise_num = exercise_match.group(1)
                        tema = exercise_match.group(2)
                        
                        retrieved_docs = [doc for doc in all_docs if 
                                         doc.metadata.get('exercise_number') == exercise_num or
                                         (doc.metadata.get('tema', '').startswith(tema) and 
                                          f"Ejercicio {exercise_num}" in doc.page_content)]
        
        # Formatear contexto
        context = self._format_context(retrieved_docs)
        
        # Si está habilitada la memoria contextual
        if use_memory and (self.conversation_history or conversation_context):
            context_parts = []
            
            # Añadir contexto de la conversación actual si existe
            if conversation_context:
                context_parts.append(f"Contexto actual de la conversación: {conversation_context}")
                
            # Añadir historial reciente
            if self.conversation_history:
                history_context = "\n".join([f"Usuario: {q}\nAsistente: {a}" for q, a in self.conversation_history[-3:]])
                context_parts.append(f"Conversación reciente:\n{history_context}")
            
            if context_parts:
                context += "\n\n" + "\n\n".join(context_parts)
        
        # Si estamos verificando una respuesta del usuario
        is_answer_verification = False
        if isinstance(conversation_context, dict) and conversation_context.get("query_type") == "answer_verification":
            is_answer_verification = True
            exercise_num = conversation_context.get("last_exercise")
            tema = conversation_context.get("last_tema")
            user_answer = conversation_context.get("user_answer")
            
            # Añadir instrucción especial al contexto
            verification_instruction = f"\nEl usuario ha indicado que la respuesta al ejercicio {exercise_num} del tema {tema} es la opción '{user_answer}'. Verifica si esta respuesta es correcta según la información disponible. Si no puedes verificarlo, indícalo claramente. Si la respuesta es incorrecta, NO des la respuesta correcta directamente, sino guía al usuario hacia ella."
            context += verification_instruction
        
        # Construir la cadena de prompt
        chain = (
            {"context": lambda x: context, "question": lambda x: x}
            | self.prompt_template
            | llm_to_use
            | StrOutputParser()
        )
        
        # Generar respuesta
        response = chain.invoke(query)
        
        # Estimación básica de uso de tokens
        estimated_tokens = len(query) // 3 + len(context) // 3 + len(response) // 3
        
        if model_name not in self.token_usage["by_model"]:
            self.token_usage["by_model"][model_name] = 0
        
        self.token_usage["by_model"][model_name] += estimated_tokens
        self.token_usage["total"] += estimated_tokens
        
        # Guardar en historial
        self.conversation_history.append((query, response))
        if len(self.conversation_history) > 10:  # Limitar tamaño
            self.conversation_history.pop(0)
        
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
        try:
            metadatas = self.vector_store.get(include=["metadatas"]).get("metadatas", [])
            return sorted(list(set(os.path.basename(m.get('source', 'unknown')) 
                               for m in metadatas if m.get('source'))))
        except Exception as e:
            print(f"⚠️ Error al obtener fuentes: {e}")
            return []

    
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
                        description="El número completo del tema, incluyendo subtema si existe (ej: '1', '2-2', '3-1'). Usar para búsquedas exactas como 'tema 2-2'.", 
                        type="string"
                    ),
                    AttributeInfo(
                        name="tema_principal", 
                        description="El número principal del tema, sin subtema (ej: '1', '2', '3'). Usar cuando la consulta solo menciona 'tema X' sin especificar subtema.", 
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
        # Detectar consultas sobre listas de ejercicios
        list_exercises_match = re.search(r'(qu[eé]\s+ejercicios|cu[aá]les\s+ejercicios|muéstrame\s+(?:todos\s+)?los\s+ejercicios|hay\s+(?:algún|algunos)\s+ejercicios?).*tema\s+(\d+)(?:-(\d+))?', query.lower())
        
        if list_exercises_match:
            tema_principal = list_exercises_match.group(2)
            tema_secundario = list_exercises_match.group(3)
            tema_str = f"{tema_principal}" if not tema_secundario else f"{tema_principal}-{tema_secundario}"
            
            # Devolver una señal especial para manejar esto en answer_question
            return f"list_exercises:{tema_str}", None
            
        # Código existente para temas y ejercicios específicos...
        tema_match = re.search(r'tema\s+(\d+)(?:-(\d+))?', query.lower())
        ejercicio_match = re.search(r'ejercicio\s+(\d+)|problema\s+(\d+)', query.lower())
        
        filter_parts = []
        if tema_match:
            tema_principal = tema_match.group(1)
            tema_secundario = tema_match.group(2)
            
            if tema_secundario:  # Si especificaron tema secundario (ej: "tema 3-2")
                tema_str = f"{tema_principal}-{tema_secundario}"
                filter_parts.append(f"tema == '{tema_str}'")
            else:  # Si solo especificaron tema principal (ej: "tema 3")
                # Buscar coincidencias en tema_principal
                filter_parts.append(f"tema_principal == '{tema_principal}'")
    
        if ejercicio_match:
            ejercicio = ejercicio_match.group(1) or ejercicio_match.group(2)
            filter_parts.append(f"exercise_number == '{ejercicio}'")
    
        return query, " AND ".join(filter_parts) if filter_parts else None

    def _preprocess_query_with_context(self, query, previous_messages=None):
        """Analiza la consulta considerando el contexto de mensajes anteriores."""
        # Procesamiento básico
        processed_query, filter_str = self._preprocess_query(query)
        conversation_context = None
        
        # Detectar si el usuario está respondiendo a una pregunta anterior
        answer_match = re.search(r'(la respuesta|yo digo|creo que|pienso que|diría que).*?(es|sería)?\s+(?:la\s+)?([a-d])', query.lower())
        if answer_match:
            answer_option = answer_match.group(3).strip()
            
            if not previous_messages or len(previous_messages) < 2:
                return query, filter_str, None
                
            # Buscar el último ejercicio mencionado
            last_exercise = None
            last_tema = None
            
            for msg in reversed(previous_messages):
                if msg["role"] == "assistant":
                    # Extraer ejercicio y tema
                    exercise_match = re.search(r'[Ee]jercicio\s+(\d+)\s+del\s+tema\s+(\d+)(?:-(\d+))?', msg["content"])
                    if exercise_match:
                        last_exercise = exercise_match.group(1)
                        tema_principal = exercise_match.group(2)
                        tema_secundario = exercise_match.group(3) or ""
                        last_tema = f"{tema_principal}" if not tema_secundario else f"{tema_principal}-{tema_secundario}"
                        break
                        
            if last_exercise and last_tema:
                # Crear contexto para la respuesta del usuario
                conversation_context = {
                    "last_exercise": last_exercise,
                    "last_tema": last_tema,
                    "user_answer": answer_option,
                    "query_type": "answer_verification"
                }
                
                # Crear filtro específico para este ejercicio
                exercise_filter = f"exercise_number == '{last_exercise}' AND tema == '{last_tema}'"
                
                return f"Verifica si la opción '{answer_option}' es la respuesta correcta al ejercicio {last_exercise} del tema {last_tema}", exercise_filter, conversation_context
        
        return query, filter_str, None

    def build_topic_index(self):
        """Crea un índice de temas y ejercicios disponibles para referencia rápida."""
        print("🔄 Construyendo índice de temas y ejercicios...")
        self.topic_index = {}
        
        if not self.vector_store:
            print("⚠️ No se puede construir índice de temas: Vector store no inicializado")
            return self.topic_index
            
        try:
            # Usar colección directamente para obtener metadatos
            collection = self.vector_store._collection
            metadatas = collection.get(include=["metadatas"])["metadatas"]
            
            if not metadatas:
                print("⚠️ No se encontraron metadatos en la colección")
                return self.topic_index
                
            print(f"📊 Analizando {len(metadatas)} documentos para construir índice...")
            
            # Primer paso: Recopilar todos los temas y ejercicios
            temas_principales = set()
            temas_completos = set()
            for meta in metadatas:
                if 'tema' in meta:
                    temas_completos.add(meta['tema'])
                    if 'tema_principal' in meta:
                        temas_principales.add(meta['tema_principal'])
            
            # Segundo paso: Para cada tema, obtener todos sus ejercicios
            for tema in temas_completos:
                self.topic_index[tema] = set()
                
                # Extraer ejercicios específicos de este tema
                for meta in metadatas:
                    if meta.get('tema') == tema and 'exercise_number' in meta:
                        self.topic_index[tema].add(meta['exercise_number'])
            
            # Tercer paso: Agregar referencias cruzadas para temas principales
            for tema_principal in temas_principales:
                # Identificar todos los subtemas
                subtemas = [t for t in temas_completos if t.startswith(f"{tema_principal}-")]
                
                # Si hay subtemas y el tema principal no está ya en el índice
                if subtemas and tema_principal not in self.topic_index:
                    self.topic_index[tema_principal] = set()
                    
                    # Copiar ejercicios de todos los subtemas
                    for subtema in subtemas:
                        if subtema in self.topic_index:
                            self.topic_index[tema_principal].update(self.topic_index[subtema])
            
            # Verificar resultados
            print(f"✅ Índice de temas construido: {len(self.topic_index)} temas encontrados")
            for tema, ejercicios in sorted(self.topic_index.items()):
                print(f"  - Tema {tema}: {len(ejercicios)} ejercicios → {', '.join(sorted(ejercicios, key=lambda x: int(x)))}")
                
        except Exception as e:
            print(f"❌ Error al construir índice de temas: {e}")
            traceback.print_exc()
        
        return self.topic_index

    def list_available_topics(self):
        """Devuelve una lista formateada de temas y ejercicios disponibles."""
        if not hasattr(self, 'topic_index') or not self.topic_index:
            self.build_topic_index()
        
        if not self.topic_index:
            return "No se encontraron temas o ejercicios indexados. Asegúrate de que los documentos estén correctamente cargados."
        
        result = "Temas y ejercicios disponibles:\n\n"
        
        # Primero mostrar temas principales
        temas_principales = sorted([t for t in self.topic_index.keys() if "-" not in t], 
                                   key=lambda x: int(x) if x.isdigit() else 0)
        
        for tema in temas_principales:
            ejercicios = sorted(list(self.topic_index[tema]), key=lambda x: int(x))
            result += f"Tema {tema}: Ejercicios {', '.join(ejercicios)}\n"
        
        # Luego mostrar subtemas
        result += "\nSubtemas:\n"
        subtemas = sorted([t for t in self.topic_index.keys() if "-" in t],
                          key=lambda x: [int(p) if p.isdigit() else 0 for p in x.split("-")])
        
        for tema in subtemas:
            ejercicios = sorted(list(self.topic_index[tema]), key=lambda x: int(x))
            result += f"Tema {tema}: Ejercicios {', '.join(ejercicios)}\n"
        
        return result

    def load_existing_data(self, directory_path):
        """Carga datos existentes desde un directorio previamente procesado."""
        print(f"📂 Cargando datos existentes desde: {directory_path}")
        return self.load_and_index_preprocessed(directory_path)

    def get_exercises_by_topic(self, tema):
        """
        Obtiene todos los ejercicios disponibles para un tema específico.
        
        Args:
            tema (str): El número del tema (ej: "1", "2-2", "3")
            
        Returns:
            str: Una respuesta formateada con la lista de ejercicios
        """
        # Asegurarse que el índice está construido
        if not hasattr(self, 'topic_index') or not self.topic_index:
            self.build_topic_index()
        
        # Buscar tema exacto
        if tema in self.topic_index:
            ejercicios = sorted(list(self.topic_index[tema]), key=lambda x: int(x))
            
            if not ejercicios:
                return f"No se encontraron ejercicios para el tema {tema}."
                
            # Construir respuesta
            response = f"Los ejercicios disponibles para el tema {tema} son:\n\n"
            for ej in ejercicios:
                response += f"- Ejercicio {ej}\n"
                
            return response
        
        # Buscar tema principal si es que hay subtemas
        if "-" not in tema:
            # Encontrar subtemas
            subtemas = [t for t in self.topic_index.keys() if t.startswith(f"{tema}-")]
            
            if subtemas:
                response = f"Para el tema {tema}, tengo información sobre los siguientes subtemas:\n\n"
                
                for subtema in sorted(subtemas):
                    ejercicios = sorted(list(self.topic_index[subtema]), key=lambda x: int(x))
                    response += f"Tema {subtema}: Ejercicios {', '.join(ejercicios)}\n"
                    
                return response
        
        return f"No encontré información sobre ejercicios del tema {tema}."