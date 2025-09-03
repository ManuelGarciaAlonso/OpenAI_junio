import json
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import os
from dotenv import load_dotenv
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import random
from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

class NoRAGEvaluator:
    def __init__(self, api_key, corpus_path, models_to_test=None):
        """
        Inicializa el evaluador sin RAG para comparación directa.
        
        Args:
            api_key: API key de OpenAI
            corpus_path: Ruta al archivo JSON con el corpus de evaluación
            models_to_test: Lista de modelos a evaluar
        """
        self.api_key = api_key
        self.corpus_path = corpus_path
        self.models_to_test = models_to_test or ["gpt-4o", "o4-mini", "gpt-5-mini"]
        
        # Cargar corpus
        with open(corpus_path, 'r', encoding='utf-8') as f:
            self.corpus = json.load(f)
            
        # Preparar métricas
        self.metrics = defaultdict(dict)
        self.results = defaultdict(list)
        
        # Inicializar modelos de evaluación
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.semantic_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        
        # Crear identificador único para la sesión de evaluación
        self.eval_id = f"norag_eval_{time.strftime('%Y%m%d_%H%M%S')}"
        
        # Configuración de rate limiting
        self.min_delay_between_requests = 2.0
        self.rate_limit_backoff = 30.0
        
        # Inicializar LLMs
        self._initialize_llms()
        
        # Crear prompt para evaluación sin RAG
        self._setup_prompt()
        
        print(f"✅ Evaluador NoRAG inicializado con {len(self.corpus)} preguntas.")
        print(f"🔍 Modelos a evaluar: {self.models_to_test}")
    
    def _initialize_llms(self):
        """Inicializa los modelos LLM para evaluación sin RAG."""
        self.llm_instances = {}
        
        for model_name in self.models_to_test:
            try:
                if model_name in ["o1-preview", "o1-mini"]:
                    # Modelos que usan Completion API
                    llm = OpenAI(
                        model=model_name,
                        openai_api_key=self.api_key,
                        temperature=1
                    )
                else:
                    # Modelos que usan Chat API
                    llm = ChatOpenAI(
                        model=model_name,
                        openai_api_key=self.api_key,
                        temperature=1
                    )
                
                self.llm_instances[model_name] = llm
                print(f"✅ Modelo {model_name} inicializado correctamente.")
                
            except Exception as e:
                print(f"❌ Error inicializando modelo {model_name}: {e}")
    
    def _setup_prompt(self):
        """Configura el prompt para evaluación sin RAG."""
        self.prompt_template = PromptTemplate.from_template(
            """Eres un asistente experto en Fundamentos del Software. Responde a la siguiente pregunta basándote únicamente en tu conocimiento previo sobre el tema, sin acceso a materiales específicos del curso.

Pregunta: {question}

Instrucciones:
1. Proporciona una respuesta clara y pedagógica
2. Si no estás seguro de algún detalle específico, indícalo
3. Mantén un enfoque educativo apropiado para estudiantes universitarios
4. No inventes información específica que no conozcas con certeza

Respuesta:"""
        )
    
    def _handle_api_request_with_retry(self, question, model_name, max_retries=3):
        """
        Maneja las requests a la API con retry automático.
        
        Args:
            question: La pregunta a enviar
            model_name: Nombre del modelo
            max_retries: Número máximo de reintentos
            
        Returns:
            tuple: (response, response_time) o (None, 0) si falla
        """
        llm = self.llm_instances.get(model_name)
        if not llm:
            raise ValueError(f"Modelo {model_name} no disponible")
        
        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()
                
                # Crear la cadena de procesamiento
                if model_name in ["o1-preview", "o1-mini"]:
                    # Para modelos de Completion
                    prompt_text = self.prompt_template.format(question=question)
                    response = llm.invoke(prompt_text)
                else:
                    # Para modelos de Chat
                    chain = self.prompt_template | llm | StrOutputParser()
                    response = chain.invoke({"question": question})
                
                response_time = time.time() - start_time
                return response, response_time
                
            except Exception as e:
                error_str = str(e)
                
                # Rate limit error
                if "rate_limit_exceeded" in error_str or "429" in error_str:
                    if attempt < max_retries:
                        wait_time = self.rate_limit_backoff * (2 ** attempt)
                        print(f"⏳ Rate limit. Esperando {wait_time:.1f}s (intento {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise e
                
                # Otros errores
                else:
                    if attempt < max_retries:
                        wait_time = 5 * (attempt + 1)
                        print(f"⚠️ Error en intento {attempt + 1}: {error_str}. Reintentando en {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise e
        
        return None, 0
    
    def run_evaluation(self, n_samples=None, categories=None, difficulties=None, random_seed=42):
        """
        Ejecuta la evaluación sin RAG en el corpus.
        """
        # Filtrar corpus según parámetros
        filtered_corpus = self.corpus
        
        if categories:
            filtered_corpus = [q for q in filtered_corpus if q.get('category') in categories]
        if difficulties:
            filtered_corpus = [q for q in filtered_corpus if q.get('difficulty') in difficulties]
        
        # Seleccionar muestras
        if n_samples and n_samples < len(filtered_corpus):
            random.seed(random_seed)
            filtered_corpus = random.sample(filtered_corpus, n_samples)
        
        print(f"🚀 Iniciando evaluación SIN RAG con {len(filtered_corpus)} preguntas...")
        
        # Para cada modelo
        for model_name in self.models_to_test:
            if model_name not in self.llm_instances:
                print(f"⚠️ Modelo {model_name} no disponible. Saltando...")
                continue
                
            print(f"\n📝 Evaluando modelo SIN RAG: {model_name}")
            model_results = []
            
            # Evaluar cada pregunta
            for idx, question_obj in enumerate(tqdm(filtered_corpus, desc=f"NoRAG-{model_name}")):
                question = question_obj.get('question', '')
                ground_truth = question_obj.get('ground_truth', '')
                category = question_obj.get('category', 'sin_categoria')
                difficulty = question_obj.get('difficulty', 'sin_dificultad')
                metadata = question_obj.get('metadata', {})
                
                try:
                    # Obtener respuesta del LLM sin RAG
                    response, response_time = self._handle_api_request_with_retry(
                        question, model_name
                    )
                    
                    if response is None:
                        raise Exception("Failed after all retries")
                    
                    # Evaluar la respuesta
                    eval_scores = self._evaluate_response(response, ground_truth)
                    
                    # Registrar resultado
                    result = {
                        'question_id': idx,
                        'question': question,
                        'ground_truth': ground_truth,
                        'response': response,
                        'category': category,
                        'difficulty': difficulty,
                        'metadata': metadata,
                        'response_time': response_time,
                        'retrieved_docs': 0,  # Sin RAG = 0 documentos
                        'rag_used': False,    # Indicador para comparación
                        **eval_scores
                    }
                    model_results.append(result)
                    
                    # Delay entre requests
                    time.sleep(self.min_delay_between_requests)
                    
                except Exception as e:
                    print(f"\n❌ Error evaluando pregunta #{idx}: {e}")
                    # Registrar error
                    result = {
                        'question_id': idx,
                        'question': question,
                        'ground_truth': ground_truth,
                        'response': f"ERROR: {str(e)}",
                        'category': category,
                        'difficulty': difficulty,
                        'metadata': metadata,
                        'response_time': 0,
                        'retrieved_docs': 0,
                        'rag_used': False,
                        'error': str(e),
                        'rouge1_f': 0,
                        'rouge2_f': 0,
                        'rougeL_f': 0,
                        'semantic_similarity': 0,
                    }
                    model_results.append(result)
                    
                    time.sleep(5)
            
            # Guardar resultados del modelo
            self.results[model_name] = model_results
            
            # Calcular métricas agregadas
            self._calculate_aggregated_metrics(model_name)
            
        print("✅ Evaluación SIN RAG completada.")
    
    def _evaluate_response(self, response, ground_truth):
        """Evalúa la respuesta usando las mismas métricas que el evaluador RAG."""
        try:
            # ROUGE scores
            rouge_scores = self.rouge_scorer.score(ground_truth, response)
            
            # Semantic similarity
            ground_truth_emb = self.semantic_model.encode(ground_truth)
            response_emb = self.semantic_model.encode(response)
            semantic_similarity = util.pytorch_cos_sim(ground_truth_emb, response_emb).item()
            
            return {
                'rouge1_f': rouge_scores['rouge1'].fmeasure,
                'rouge2_f': rouge_scores['rouge2'].fmeasure,
                'rougeL_f': rouge_scores['rougeL'].fmeasure,
                'semantic_similarity': semantic_similarity
            }
        except Exception as e:
            print(f"⚠️ Error evaluando métricas: {e}")
            return {
                'rouge1_f': 0,
                'rouge2_f': 0,
                'rougeL_f': 0,
                'semantic_similarity': 0
            }
    
    def _calculate_aggregated_metrics(self, model_name):
        """Calcula métricas agregadas por categoría, dificultad, etc."""
        results = pd.DataFrame(self.results[model_name])
        
        # Filtrar errores para cálculo de métricas
        valid_results = results[results['rouge1_f'] > 0]
        
        if len(valid_results) == 0:
            print(f"⚠️ No hay resultados válidos para el modelo {model_name}")
            return
        
        # Métricas globales
        self.metrics[model_name]['global'] = {
            'rouge1_f': valid_results['rouge1_f'].mean(),
            'rouge2_f': valid_results['rouge2_f'].mean(),
            'rougeL_f': valid_results['rougeL_f'].mean(),
            'semantic_similarity': valid_results['semantic_similarity'].mean(),
            'avg_response_time': valid_results['response_time'].mean(),
            'total_questions': len(results),
            'successful_questions': len(valid_results),
            'error_rate': (len(results) - len(valid_results)) / len(results)
        }
        
        # Métricas por categoría
        self.metrics[model_name]['by_category'] = {}
        for category in valid_results['category'].unique():
            cat_data = valid_results[valid_results['category'] == category]
            if len(cat_data) > 0:
                self.metrics[model_name]['by_category'][category] = {
                    'rouge1_f': cat_data['rouge1_f'].mean(),
                    'rouge2_f': cat_data['rouge2_f'].mean(),
                    'rougeL_f': cat_data['rougeL_f'].mean(),
                    'semantic_similarity': cat_data['semantic_similarity'].mean(),
                    'count': len(cat_data)
                }
        
        # Métricas por dificultad
        self.metrics[model_name]['by_difficulty'] = {}
        for difficulty in valid_results['difficulty'].unique():
            diff_data = valid_results[valid_results['difficulty'] == difficulty]
            if len(diff_data) > 0:
                self.metrics[model_name]['by_difficulty'][difficulty] = {
                    'rouge1_f': diff_data['rouge1_f'].mean(),
                    'rouge2_f': diff_data['rouge2_f'].mean(),
                    'rougeL_f': diff_data['rougeL_f'].mean(),
                    'semantic_similarity': diff_data['semantic_similarity'].mean(),
                    'count': len(diff_data)
                }
    
    def save_results(self, output_dir="evaluation_results"):
        """
        Guarda los resultados de la evaluación sin RAG.
        """
        # Crear carpeta principal
        base_output_dir = os.path.join(output_dir, self.eval_id)
        os.makedirs(base_output_dir, exist_ok=True)
        
        # Subcarpetas
        metrics_dir = os.path.join(base_output_dir, "metrics")
        examples_dir = os.path.join(base_output_dir, "examples")
        data_dir = os.path.join(base_output_dir, "data")
        
        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(examples_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        
        # Guardar métricas
        with open(os.path.join(metrics_dir, "metrics_norag.json"), 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        
        # Guardar resultados detallados
        for model_name, results in self.results.items():
            model_dir = os.path.join(data_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            
            df = pd.DataFrame(results)
            df.to_csv(os.path.join(model_dir, "results_norag.csv"), index=False, encoding='utf-8')
            
            # Ejemplos de respuestas sin RAG vs con RAG esperado
            df_sorted = df.sort_values(by='semantic_similarity', ascending=False)
            best_examples = df_sorted.head(5)
            worst_examples = df_sorted.tail(5)
            
            with open(os.path.join(examples_dir, f"{model_name}_norag_examples.md"), 'w', encoding='utf-8') as f:
                f.write(f"# Ejemplos Sin RAG - Modelo {model_name}\n\n")
                
                f.write("## Mejores Respuestas Sin RAG\n\n")
                for _, example in best_examples.iterrows():
                    f.write(f"### Pregunta: {example['question']}\n\n")
                    f.write(f"**Ground Truth (con materiales):**\n{example['ground_truth']}\n\n")
                    f.write(f"**Respuesta Sin RAG:**\n{example['response']}\n\n")
                    f.write(f"**Métricas:** Similitud: {example['semantic_similarity']:.4f}, Rouge-L: {example['rougeL_f']:.4f}\n\n")
                    f.write("---\n\n")
                
                f.write("## Peores Respuestas Sin RAG\n\n")
                for _, example in worst_examples.iterrows():
                    f.write(f"### Pregunta: {example['question']}\n\n")
                    f.write(f"**Ground Truth (con materiales):**\n{example['ground_truth']}\n\n")
                    f.write(f"**Respuesta Sin RAG:**\n{example['response']}\n\n")
                    f.write(f"**Métricas:** Similitud: {example['semantic_similarity']:.4f}, Rouge-L: {example['rougeL_f']:.4f}\n\n")
                    f.write("---\n\n")
        
        # Metadatos
        with open(os.path.join(base_output_dir, "evaluation_info.json"), 'w', encoding='utf-8') as f:
            info = {
                "eval_id": self.eval_id,
                "evaluation_type": "NO_RAG",
                "corpus_path": self.corpus_path,
                "models_evaluated": self.models_to_test,
                "total_questions": len(self.corpus),
                "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Resultados sin RAG guardados en '{base_output_dir}'")
        return base_output_dir
    
    def run_full_evaluation(self, n_samples=None, output_dir="evaluation_results"):
        """
        Ejecuta la evaluación completa sin RAG.
        """
        # Ejecutar evaluación
        self.run_evaluation(n_samples=n_samples)
        
        # Guardar resultados
        results_dir = self.save_results(output_dir)
        
        return results_dir

# Comparador para analizar diferencias RAG vs NoRAG
class RAGComparator:
    def __init__(self, rag_results_dir, norag_results_dir):
        """
        Inicializa el comparador entre resultados RAG y NoRAG.
        
        Args:
            rag_results_dir: Directorio con resultados del evaluador RAG
            norag_results_dir: Directorio con resultados del evaluador NoRAG
        """
        self.rag_results_dir = rag_results_dir
        self.norag_results_dir = norag_results_dir
        
        # Cargar datos
        self._load_data()
    
    def _load_data(self):
        """Carga los datos de ambas evaluaciones."""
        # Cargar métricas RAG
        rag_metrics_path = os.path.join(self.rag_results_dir, "metrics", "metrics.json")
        with open(rag_metrics_path, 'r', encoding='utf-8') as f:
            self.rag_metrics = json.load(f)
        
        # Cargar métricas NoRAG
        norag_metrics_path = os.path.join(self.norag_results_dir, "metrics", "metrics_norag.json")
        with open(norag_metrics_path, 'r', encoding='utf-8') as f:
            self.norag_metrics = json.load(f)
        
        # Cargar datos detallados para análisis
        self.rag_data = {}
        self.norag_data = {}
        
        # Buscar modelos comunes
        common_models = set(self.rag_metrics.keys()) & set(self.norag_metrics.keys())
        
        for model in common_models:
            # Cargar datos RAG
            rag_csv_path = os.path.join(self.rag_results_dir, "data", model, "results.csv")
            if os.path.exists(rag_csv_path):
                self.rag_data[model] = pd.read_csv(rag_csv_path)
            
            # Cargar datos NoRAG
            norag_csv_path = os.path.join(self.norag_results_dir, "data", model, "results_norag.csv")
            if os.path.exists(norag_csv_path):
                self.norag_data[model] = pd.read_csv(norag_csv_path)
        
        print(f"✅ Datos cargados para modelos: {list(common_models)}")
    
    def generate_comparison_report(self, output_file="rag_vs_norag_comparison.md"):
        """
        Genera un informe comparativo entre RAG y NoRAG.
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Comparación RAG vs No-RAG\n\n")
            f.write(f"Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Comparativa global
            f.write("## Comparativa Global de Métricas\n\n")
            
            table_data = []
            headers = ["Modelo", "Método", "Rouge-1", "Rouge-2", "Rouge-L", "Similitud Semántica", "Tiempo (s)"]
            
            common_models = set(self.rag_metrics.keys()) & set(self.norag_metrics.keys())
            
            for model in common_models:
                # RAG
                rag_global = self.rag_metrics[model]['global']
                table_data.append([
                    model, "CON RAG",
                    f"{rag_global['rouge1_f']:.4f}",
                    f"{rag_global['rouge2_f']:.4f}",
                    f"{rag_global['rougeL_f']:.4f}",
                    f"{rag_global['semantic_similarity']:.4f}",
                    f"{rag_global['avg_response_time']:.2f}"
                ])
                
                # NoRAG
                norag_global = self.norag_metrics[model]['global']
                table_data.append([
                    model, "SIN RAG",
                    f"{norag_global['rouge1_f']:.4f}",
                    f"{norag_global['rouge2_f']:.4f}",
                    f"{norag_global['rougeL_f']:.4f}",
                    f"{norag_global['semantic_similarity']:.4f}",
                    f"{norag_global['avg_response_time']:.2f}"
                ])
                
                # Diferencia
                diff_sim = rag_global['semantic_similarity'] - norag_global['semantic_similarity']
                diff_time = rag_global['avg_response_time'] - norag_global['avg_response_time']
                
                table_data.append([
                    model, "DIFERENCIA",
                    f"{rag_global['rouge1_f'] - norag_global['rouge1_f']:+.4f}",
                    f"{rag_global['rouge2_f'] - norag_global['rouge2_f']:+.4f}",
                    f"{rag_global['rougeL_f'] - norag_global['rougeL_f']:+.4f}",
                    f"{diff_sim:+.4f}",
                    f"{diff_time:+.2f}"
                ])
                
                table_data.append(["", "", "", "", "", "", ""])  # Separador
            
            f.write(tabulate(table_data, headers=headers, tablefmt="pipe"))
            f.write("\n\n")
            
            # Análisis de casos donde RAG "sobre-responde"
            f.write("## Análisis de Sobre-Respuesta del RAG\n\n")
            
            for model in common_models:
                if model in self.rag_data and model in self.norag_data:
                    rag_df = self.rag_data[model]
                    norag_df = self.norag_data[model]
                    
                    # Encontrar casos donde NoRAG es mejor que RAG
                    merged_df = pd.merge(rag_df, norag_df, on='question_id', suffixes=('_rag', '_norag'))
                    
                    # Casos donde sin RAG es mejor
                    over_response_cases = merged_df[
                        merged_df['semantic_similarity_norag'] > merged_df['semantic_similarity_rag']
                    ]
                    
                    f.write(f"### Modelo: {model}\n\n")
                    f.write(f"- **Total de preguntas evaluadas:** {len(merged_df)}\n")
                    f.write(f"- **Casos donde Sin RAG > Con RAG:** {len(over_response_cases)} ({len(over_response_cases)/len(merged_df)*100:.1f}%)\n")
                    
                    if len(over_response_cases) > 0:
                        f.write(f"- **Diferencia promedio en estos casos:** {(over_response_cases['semantic_similarity_norag'] - over_response_cases['semantic_similarity_rag']).mean():.4f}\n")
                        
                        # Top 3 casos donde sin RAG es significativamente mejor
                        top_cases = over_response_cases.nlargest(3, 'semantic_similarity_norag')
                        
                        f.write(f"\n**Top 3 casos donde Sin RAG supera significativamente a RAG:**\n\n")
                        
                        for idx, case in top_cases.iterrows():
                            f.write(f"**Pregunta:** {case['question_rag']}\n\n")
                            f.write(f"**Ground Truth:** {case['ground_truth_rag']}\n\n")
                            f.write(f"**Respuesta RAG:** {case['response_rag'][:200]}...\n\n")
                            f.write(f"**Respuesta Sin RAG:** {case['response_norag'][:200]}...\n\n")
                            f.write(f"**Similitud RAG:** {case['semantic_similarity_rag']:.4f}\n")
                            f.write(f"**Similitud Sin RAG:** {case['semantic_similarity_norag']:.4f}\n\n")
                            f.write("---\n\n")
            
            f.write("## Conclusiones\n\n")
            f.write("1. **Impacto del RAG**: Compare las métricas globales para ver si el RAG mejora o perjudica el rendimiento general.\n")
            f.write("2. **Casos de sobre-respuesta**: Analice los casos donde Sin RAG supera a RAG para identificar patrones.\n")
            f.write("3. **Eficiencia**: Compare los tiempos de respuesta para evaluar el costo computacional del RAG.\n")
        
        print(f"✅ Informe comparativo generado: {output_file}")

# Script principal
if __name__ == "__main__":
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    corpus_path = "preguntas_fs.json"
    models_to_test = ["gpt-4o", "o4-mini", "gpt-5-mini"]  # Modelos comunes
    
    try:
        print("🚀 Ejecutando evaluación SIN RAG...")
        
        # Crear evaluador sin RAG
        norag_evaluator = NoRAGEvaluator(api_key, corpus_path, models_to_test)
        
        norag_results_dir = norag_evaluator.run_full_evaluation()
        
        print(f"✅ Evaluación sin RAG completada: {norag_results_dir}")
        
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()