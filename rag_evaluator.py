import json
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from rag import PDFChatbot
import os
from dotenv import load_dotenv
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

class RAGEvaluator:
    def __init__(self, rag_chatbot, corpus_path, models_to_test=None):
        """
        Inicializa el evaluador de RAG.
        
        Args:
            rag_chatbot: Instancia de PDFChatbot inicializada
            corpus_path: Ruta al archivo JSON con el corpus de evaluación
            models_to_test: Lista de modelos a evaluar (si es None, se usará el modelo por defecto)
        """
        self.chatbot = rag_chatbot
        self.corpus_path = corpus_path
        self.models_to_test = models_to_test or [list(self.chatbot.llm_instances.keys())[0]]
        
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
        self.eval_id = f"eval_{time.strftime('%Y%m%d_%H%M%S')}"
        
        # Extraer categorías y dificultades disponibles
        categories = set()
        difficulties = set()
        for q in self.corpus:
            if 'category' in q:
                categories.add(q['category'])
            if 'difficulty' in q:
                difficulties.add(q['difficulty'])
        
        print(f"✅ Evaluador inicializado con {len(self.corpus)} preguntas.")
        print(f"📊 Categorías: {categories}")
        print(f"📊 Dificultades: {difficulties}")
        print(f"🔍 Modelos a evaluar: {self.models_to_test}")
    
    def run_evaluation(self, n_samples=None, categories=None, difficulties=None, random_seed=42):
        """
        Ejecuta la evaluación en el corpus.
        
        Args:
            n_samples: Número de preguntas a evaluar (si es None, se evaluarán todas)
            categories: Lista de categorías a incluir (si es None, se incluyen todas)
            difficulties: Lista de dificultades a incluir (si es None, se incluyen todas)
            random_seed: Semilla para selección aleatoria de muestras
        """
        # Filtrar corpus según parámetros
        filtered_corpus = self.corpus
        
        if categories:
            filtered_corpus = [q for q in filtered_corpus if q.get('category') in categories]
        if difficulties:
            filtered_corpus = [q for q in filtered_corpus if q.get('difficulty') in difficulties]
        
        # Seleccionar muestras
        if n_samples and n_samples < len(filtered_corpus):
            np.random.seed(random_seed)
            filtered_corpus = np.random.choice(filtered_corpus, n_samples, replace=False).tolist()
        
        print(f"🚀 Iniciando evaluación con {len(filtered_corpus)} preguntas...")
        
        # Para cada modelo
        for model_name in self.models_to_test:
            print(f"\n📝 Evaluando modelo: {model_name}")
            model_results = []
            
            # Evaluar cada pregunta
            for idx, question_obj in enumerate(tqdm(filtered_corpus, desc=f"Evaluando con {model_name}")):
                question = question_obj.get('question', '')
                ground_truth = question_obj.get('ground_truth', '')
                category = question_obj.get('category', 'sin_categoria')
                difficulty = question_obj.get('difficulty', 'sin_dificultad')
                metadata = question_obj.get('metadata', {})
                
                try:
                    # Obtener respuesta del chatbot
                    start_time = time.time()
                    response, docs = self.chatbot.answer_question(question, model_name=model_name)
                    response_time = time.time() - start_time
                    
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
                        'retrieved_docs': len(docs),
                        **eval_scores
                    }
                    model_results.append(result)
                    
                    # Dar tiempo para no saturar la API
                    time.sleep(0.5)
                    
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
                        'error': str(e),
                        'rouge1_f': 0,
                        'rouge2_f': 0,
                        'rougeL_f': 0,
                        'semantic_similarity': 0,
                    }
                    model_results.append(result)
            
            # Guardar resultados del modelo
            self.results[model_name] = model_results
            
            # Calcular métricas agregadas
            self._calculate_aggregated_metrics(model_name)
            
        print("✅ Evaluación completada.")
    
    def _evaluate_response(self, response, ground_truth):
        """Evalúa la respuesta comparándola con el ground truth usando varias métricas."""
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
    
    def _calculate_aggregated_metrics(self, model_name):
        """Calcula métricas agregadas por categoría, dificultad, etc."""
        results = pd.DataFrame(self.results[model_name])
        
        # Métricas globales
        self.metrics[model_name]['global'] = {
            'rouge1_f': results['rouge1_f'].mean(),
            'rouge2_f': results['rouge2_f'].mean(),
            'rougeL_f': results['rougeL_f'].mean(),
            'semantic_similarity': results['semantic_similarity'].mean(),
            'avg_response_time': results['response_time'].mean(),
            'total_questions': len(results)
        }
        
        # Métricas por categoría
        self.metrics[model_name]['by_category'] = {}
        for category in results['category'].unique():
            cat_data = results[results['category'] == category]
            self.metrics[model_name]['by_category'][category] = {
                'rouge1_f': cat_data['rouge1_f'].mean(),
                'rouge2_f': cat_data['rouge2_f'].mean(),
                'rougeL_f': cat_data['rougeL_f'].mean(),
                'semantic_similarity': cat_data['semantic_similarity'].mean(),
                'count': len(cat_data)
            }
        
        # Métricas por dificultad
        self.metrics[model_name]['by_difficulty'] = {}
        for difficulty in results['difficulty'].unique():
            diff_data = results[results['difficulty'] == difficulty]
            self.metrics[model_name]['by_difficulty'][difficulty] = {
                'rouge1_f': diff_data['rouge1_f'].mean(),
                'rouge2_f': diff_data['rouge2_f'].mean(),
                'rougeL_f': diff_data['rougeL_f'].mean(),
                'semantic_similarity': diff_data['semantic_similarity'].mean(),
                'count': len(diff_data)
            }
    
    def save_results(self, output_dir="evaluation_results"):
        """
        Guarda los resultados de la evaluación en una estructura de directorios organizada.
        
        Args:
            output_dir: Directorio base donde se crearán las carpetas de resultados
            
        Returns:
            str: Ruta al directorio de resultados creado
        """
        # Crear carpeta principal con el ID único de evaluación
        base_output_dir = os.path.join(output_dir, self.eval_id)
        os.makedirs(base_output_dir, exist_ok=True)
        
        # Subcarpetas para diferentes tipos de resultados
        metrics_dir = os.path.join(base_output_dir, "metrics")
        examples_dir = os.path.join(base_output_dir, "examples")
        data_dir = os.path.join(base_output_dir, "data")
        
        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(examples_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        
        # Guardar métricas
        with open(os.path.join(metrics_dir, "metrics.json"), 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        
        # Guardar resultados detallados por modelo
        for model_name, results in self.results.items():
            # Crear subcarpeta para cada modelo
            model_dir = os.path.join(data_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            
            # Guardar resultados CSV
            df = pd.DataFrame(results)
            df.to_csv(os.path.join(model_dir, "results.csv"), index=False, encoding='utf-8')
            
            # Guardar ejemplos más relevantes (mejores y peores)
            df_sorted = df.sort_values(by='semantic_similarity', ascending=False)
            best_examples = df_sorted.head(5)
            worst_examples = df_sorted.tail(5)
            
            with open(os.path.join(examples_dir, f"{model_name}_examples.md"), 'w', encoding='utf-8') as f:
                f.write(f"# Ejemplos de Respuestas del Modelo {model_name}\n\n")
                
                f.write("## Mejores Ejemplos\n\n")
                for _, example in best_examples.iterrows():
                    f.write(f"### Pregunta: {example['question']}\n\n")
                    f.write(f"**Ground Truth:**\n{example['ground_truth']}\n\n")
                    f.write(f"**Respuesta:**\n{example['response']}\n\n")
                    f.write(f"**Métricas:** Similitud Semántica: {example['semantic_similarity']:.4f}, Rouge-L: {example['rougeL_f']:.4f}\n\n")
                    f.write("---\n\n")
                
                f.write("## Peores Ejemplos\n\n")
                for _, example in worst_examples.iterrows():
                    f.write(f"### Pregunta: {example['question']}\n\n")
                    f.write(f"**Ground Truth:**\n{example['ground_truth']}\n\n")
                    f.write(f"**Respuesta:**\n{example['response']}\n\n")
                    f.write(f"**Métricas:** Similitud Semántica: {example['semantic_similarity']:.4f}, Rouge-L: {example['rougeL_f']:.4f}\n\n")
                    f.write("---\n\n")
        
        # Guardar metadatos de la evaluación
        with open(os.path.join(base_output_dir, "evaluation_info.json"), 'w', encoding='utf-8') as f:
            info = {
                "eval_id": self.eval_id,
                "corpus_path": self.corpus_path,
                "models_evaluated": self.models_to_test,
                "total_questions": len(self.corpus),
                "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Resultados guardados en '{base_output_dir}'")
        return base_output_dir
    
    def generate_report(self, output_dir=None):
        """
        Genera un informe completo de la evaluación.
        
        Args:
            output_dir: Directorio donde se guardará el informe. Si es None, se usa el directorio
                       de la evaluación actual.
        
        Returns:
            str: Ruta al archivo de informe generado
        """
        if output_dir is None:
            output_dir = os.path.join("evaluation_results", self.eval_id)
            
        # Asegurarse que el directorio existe
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, "evaluation_report.md")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Informe de Evaluación del Asistente RAG\n\n")
            f.write(f"Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Corpus: {self.corpus_path}\n")
            f.write(f"Modelos evaluados: {', '.join(self.models_to_test)}\n\n")
            
            # Métricas globales
            f.write("## Métricas Globales\n\n")
            
            table_data = []
            headers = ["Modelo", "Rouge-1", "Rouge-2", "Rouge-L", "Similitud Semántica", "Tiempo Respuesta (s)", "Preguntas"]
            
            for model_name in self.models_to_test:
                global_metrics = self.metrics[model_name]['global']
                table_data.append([
                    model_name,
                    f"{global_metrics['rouge1_f']:.4f}",
                    f"{global_metrics['rouge2_f']:.4f}",
                    f"{global_metrics['rougeL_f']:.4f}",
                    f"{global_metrics['semantic_similarity']:.4f}",
                    f"{global_metrics['avg_response_time']:.2f}",
                    global_metrics['total_questions']
                ])
            
            f.write(tabulate(table_data, headers=headers, tablefmt="pipe"))
            f.write("\n\n")
            
            # Métricas por categoría
            f.write("## Métricas por Categoría\n\n")
            
            for model_name in self.models_to_test:
                f.write(f"### Modelo: {model_name}\n\n")
                
                table_data = []
                headers = ["Categoría", "Rouge-1", "Rouge-2", "Rouge-L", "Similitud Semántica", "Preguntas"]
                
                for category, metrics in self.metrics[model_name]['by_category'].items():
                    table_data.append([
                        category,
                        f"{metrics['rouge1_f']:.4f}",
                        f"{metrics['rouge2_f']:.4f}",
                        f"{metrics['rougeL_f']:.4f}",
                        f"{metrics['semantic_similarity']:.4f}",
                        metrics['count']
                    ])
                
                f.write(tabulate(table_data, headers=headers, tablefmt="pipe"))
                f.write("\n\n")
            
            # Métricas por dificultad
            f.write("## Métricas por Nivel de Dificultad\n\n")
            
            for model_name in self.models_to_test:
                f.write(f"### Modelo: {model_name}\n\n")
                
                table_data = []
                headers = ["Dificultad", "Rouge-1", "Rouge-2", "Rouge-L", "Similitud Semántica", "Preguntas"]
                
                for difficulty, metrics in self.metrics[model_name]['by_difficulty'].items():
                    table_data.append([
                        difficulty,
                        f"{metrics['rouge1_f']:.4f}",
                        f"{metrics['rouge2_f']:.4f}",
                        f"{metrics['rougeL_f']:.4f}",
                        f"{metrics['semantic_similarity']:.4f}",
                        metrics['count']
                    ])
                
                f.write(tabulate(table_data, headers=headers, tablefmt="pipe"))
                f.write("\n\n")
            
            f.write("## Análisis y Recomendaciones\n\n")
            
            # Análisis automático
            f.write("### Análisis de Rendimiento\n\n")
            
            # Encontrar modelo con mejor similitud semántica
            best_semantic_model = max(self.models_to_test, 
                                     key=lambda m: self.metrics[m]['global']['semantic_similarity'])
            best_semantic_score = self.metrics[best_semantic_model]['global']['semantic_similarity']
            
            f.write(f"- El modelo **{best_semantic_model}** obtuvo la mejor similitud semántica global ({best_semantic_score:.4f}).\n")
            
            # Analizar rendimiento por categoría
            for model_name in self.models_to_test:
                cat_metrics = self.metrics[model_name]['by_category']
                best_cat = max(cat_metrics.keys(), key=lambda c: cat_metrics[c]['semantic_similarity'])
                worst_cat = min(cat_metrics.keys(), key=lambda c: cat_metrics[c]['semantic_similarity'])
                
                f.write(f"- **{model_name}** rinde mejor en preguntas de tipo '**{best_cat}**' ")
                f.write(f"({cat_metrics[best_cat]['semantic_similarity']:.4f}) ")
                f.write(f"y peor en '**{worst_cat}**' ")
                f.write(f"({cat_metrics[worst_cat]['semantic_similarity']:.4f}).\n")
            
            f.write("\n### Recomendaciones\n\n")
            
            # Detectar categorías problemáticas
            problematic_categories = set()
            for model_name in self.models_to_test:
                cat_metrics = self.metrics[model_name]['by_category']
                # Identificar categorías con puntuación semántica < 0.6
                problem_cats = [c for c, m in cat_metrics.items() if m['semantic_similarity'] < 0.6]
                problematic_categories.update(problem_cats)
            
            if problematic_categories:
                f.write("Se recomienda mejorar el rendimiento en las siguientes categorías de preguntas:\n\n")
                for cat in problematic_categories:
                    f.write(f"- **{cat}**\n")
            
            # Recomendar modelo
            f.write("\n**Modelo recomendado**: ")
            f.write(f"**{best_semantic_model}** por su rendimiento superior en similitud semántica.\n")
            
            # Añadir enlaces a los ejemplos y gráficos
            f.write("\n## Enlaces a Recursos Adicionales\n\n")
            f.write("- [Ejemplos de mejores y peores respuestas](./examples/)\n")
            f.write("- [Gráficos de rendimiento](./plots/)\n")
            f.write("- [Datos detallados por modelo](./data/)\n")
        
        print(f"✅ Informe generado: {output_file}")
        return output_file
    
    def plot_metrics(self, output_dir=None):
        """
        Genera gráficos de métricas para visualizar resultados.
        
        Args:
            output_dir: Directorio donde se guardarán los gráficos. Si es None, se usa el directorio
                       de la evaluación actual.
                       
        Returns:
            str: Ruta al directorio de gráficos generado
        """
        if output_dir is None:
            output_dir = os.path.join("evaluation_results", self.eval_id)
        
        # Crear subcarpeta para gráficos
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Configuración de estilo
        sns.set_style("whitegrid")
        plt.rcParams.update({'font.size': 12})
        
        # 1. Comparativa global entre modelos
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics_data = {
            'Modelo': [],
            'Métrica': [],
            'Valor': []
        }
        
        for model_name in self.models_to_test:
            global_metrics = self.metrics[model_name]['global']
            for metric_name in ['rouge1_f', 'rouge2_f', 'rougeL_f', 'semantic_similarity']:
                metrics_data['Modelo'].append(model_name)
                metrics_data['Métrica'].append(metric_name.replace('_f', ''))
                metrics_data['Valor'].append(global_metrics[metric_name])
        
        metrics_df = pd.DataFrame(metrics_data)
        
        sns.barplot(data=metrics_df, x='Modelo', y='Valor', hue='Métrica', ax=ax)
        ax.set_title('Comparativa de Métricas por Modelo')
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "model_comparison.png"), dpi=300)
        plt.close(fig)
        
        # 2. Rendimiento por categoría para cada modelo
        for model_name in self.models_to_test:
            fig, ax = plt.subplots(figsize=(12, 7))
            
            cat_data = {
                'Categoría': [],
                'Métrica': [],
                'Valor': []
            }
            
            for cat, metrics in self.metrics[model_name]['by_category'].items():
                for metric_name in ['rouge1_f', 'rouge2_f', 'rougeL_f', 'semantic_similarity']:
                    cat_data['Categoría'].append(cat)
                    cat_data['Métrica'].append(metric_name.replace('_f', ''))
                    cat_data['Valor'].append(metrics[metric_name])
            
            cat_df = pd.DataFrame(cat_data)
            
            sns.barplot(data=cat_df, x='Categoría', y='Valor', hue='Métrica', ax=ax)
            ax.set_title(f'Rendimiento por Categoría - {model_name}')
            ax.set_ylim(0, 1)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"{model_name}_categories.png"), dpi=300)
            plt.close(fig)
        
        # 3. Rendimiento por dificultad
        for model_name in self.models_to_test:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            diff_data = {
                'Dificultad': [],
                'Métrica': [],
                'Valor': []
            }
            
            for diff, metrics in self.metrics[model_name]['by_difficulty'].items():
                for metric_name in ['rouge1_f', 'rouge2_f', 'rougeL_f', 'semantic_similarity']:
                    diff_data['Dificultad'].append(diff)
                    diff_data['Métrica'].append(metric_name.replace('_f', ''))
                    diff_data['Valor'].append(metrics[metric_name])
            
            diff_df = pd.DataFrame(diff_data)
            
            # Ordenar por dificultad si aplica
            difficulty_order = ['básica', 'intermedia', 'avanzada']
            if all(d in difficulty_order for d in diff_df['Dificultad'].unique()):
                diff_df['Dificultad'] = pd.Categorical(diff_df['Dificultad'], categories=difficulty_order, ordered=True)
                diff_df = diff_df.sort_values('Dificultad')
            
            sns.barplot(data=diff_df, x='Dificultad', y='Valor', hue='Métrica', ax=ax)
            ax.set_title(f'Rendimiento por Nivel de Dificultad - {model_name}')
            ax.set_ylim(0, 1)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"{model_name}_difficulty.png"), dpi=300)
            plt.close(fig)
            
        # 4. Gráfico de radar para comparar modelos (visualización avanzada)
        try:
            self._plot_radar_chart(plots_dir)
        except Exception as e:
            print(f"⚠️ No se pudo generar el gráfico de radar: {e}")
        
        print(f"✅ Gráficos guardados en '{plots_dir}'")
        return plots_dir
    
    def _plot_radar_chart(self, plots_dir):
        """Genera un gráfico de radar para comparar los modelos en diferentes métricas."""
        # Preparar datos
        metrics = ['rouge1_f', 'rouge2_f', 'rougeL_f', 'semantic_similarity']
        labels = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'Similitud Semántica']
        
        # Crear figura
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
        
        # Configurar ejes
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Cerrar el polígono
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(0)
        
        # Etiquetas de ejes
        plt.xticks(angles[:-1], labels)
        
        # Colores para cada modelo
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.models_to_test)))
        
        for i, model_name in enumerate(self.models_to_test):
            values = [self.metrics[model_name]['global'][m] for m in metrics]
            values += values[:1]  # Cerrar el polígono
            
            ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], label=model_name)
            ax.fill(angles, values, alpha=0.1, color=colors[i])
        
        ax.set_ylim(0, 1)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('Comparativa de Modelos por Métrica', size=15, y=1.1)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "radar_comparison.png"), dpi=300)
        plt.close(fig)

    def run_full_evaluation(self, n_samples=None, output_dir="evaluation_results"):
        """
        Ejecuta la evaluación completa y guarda todos los resultados.
        
        Args:
            n_samples: Número de muestras a evaluar (None para todas)
            output_dir: Directorio base para guardar los resultados
            
        Returns:
            str: Ruta al directorio con todos los resultados
        """
        # Ejecutar evaluación
        self.run_evaluation(n_samples=n_samples)
        
        # Guardar resultados
        results_dir = self.save_results(output_dir)
        
        # Generar informe
        self.generate_report(results_dir)
        
        # Generar gráficos
        self.plot_metrics(results_dir)
        
        # Crear archivo README para el directorio
        readme_path = os.path.join(results_dir, "README.md")
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(f"# Evaluación del Asistente RAG - {self.eval_id}\n\n")
            f.write(f"Fecha de ejecución: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## Contenido\n\n")
            f.write(f"- [Informe completo](./evaluation_report.md)\n")
            f.write(f"- [Gráficos](./plots/)\n")
            f.write(f"- [Ejemplos](./examples/)\n")
            f.write(f"- [Datos por modelo](./data/)\n")
            f.write(f"- [Métricas](./metrics/)\n\n")
            f.write(f"## Modelos evaluados\n\n")
            for model in self.models_to_test:
                f.write(f"- {model}\n")
            f.write(f"\n## Corpus\n\n")
            f.write(f"- Total de preguntas: {len(self.corpus)}\n")
            f.write(f"- Ruta: {self.corpus_path}\n")
        
        print(f"🏁 Evaluación completa guardada en: {results_dir}")
        return results_dir

# Script principal para ejecutar la evaluación
if __name__ == "__main__":
    # Cargar variables de entorno
    load_dotenv()
    
    # Inicializar chatbot
    api_key = os.getenv("OPENAI_API_KEY")
    corpus_path = "preguntas_fs.json"
    models_to_test = ["gpt-4o"]  # Puedes agregar más modelos
    
    try:
        # Verificar que el corpus existe
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"No se encontró el archivo del corpus: {corpus_path}")
        
        # Verificar que la API key está configurada
        if not api_key:
            raise ValueError("No se encontró la API key de OpenAI en las variables de entorno")
        
        # Inicializar chatbot
        print("🚀 Inicializando chatbot...")
        chatbot = PDFChatbot(
            api_key=api_key,
            chat_models=models_to_test
        )
        
        # Cargar datos si existen
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                               "Processed_Texts", "preprocessed_markdown")
        if os.path.exists(data_dir):
            print(f"📂 Cargando datos desde: {data_dir}")
            chatbot.load_existing_data(data_dir)
        else:
            raise FileNotFoundError(f"No se encontró el directorio de datos procesados: {data_dir}")
        
        # Inicializar evaluador
        print("🔍 Inicializando evaluador...")
        evaluator = RAGEvaluator(chatbot, corpus_path, models_to_test)
        
        # Ejecutar evaluación completa
        evaluator.run_full_evaluation()  # Cambiar a None para evaluar todo el corpus
        
    except Exception as e:
        print(f"❌ Error durante la evaluación: {e}")
        import traceback
        traceback.print_exc()