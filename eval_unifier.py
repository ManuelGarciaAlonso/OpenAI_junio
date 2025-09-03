import os

class EvaluationUnifier:
    def __init__(self, base_dir="evaluation_results", output_dir="unified_results"):
        """
        Inicializa el unificador de evaluaciones.
        
        Args:
            base_dir: Directorio donde se encuentran todas las carpetas de evaluación.
            output_dir: Directorio donde se guardarán los resultados unificados.
        """
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.evaluations = []
        self.rag_data = []
        self.norag_data = []
        self.unified_df = None
        
        # Crear el directorio de resultados si no existe
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Buscar todas las evaluaciones
        self._find_evaluations()
    
    # ... (resto del código sin cambios)

    def create_unified_report(self, output_file="unified_evaluation_report.md"):
        """Crea un informe unificado con todas las evaluaciones."""
        if self.unified_df is None:
            self.load_all_metrics()
        
        # Cambiar la ruta del archivo de salida
        output_file = os.path.join(self.output_dir, output_file)
        
        # Tabla global
        global_table = self.get_global_metrics_table()
        
        # Crear gráfico comparativo
        self._create_comparison_chart()
        
        # Generar informe
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Informe Unificado de Evaluaciones RAG/NoRAG\n\n")
            f.write(f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Resumen de Evaluaciones\n\n")
            
            eval_table = []
            for eval_info in self.evaluations:
                eval_table.append([
                    eval_info['eval_id'],
                    eval_info['type'],
                    eval_info['date'],
                    ", ".join(eval_info['models']),
                    eval_info['questions']
                ])
            
            f.write(tabulate(eval_table, 
                           headers=["ID Evaluación", "Tipo", "Fecha", "Modelos", "Preguntas"], 
                           tablefmt="pipe"))
            f.write("\n\n")
            
            f.write("## Métricas Globales\n\n")
            f.write(tabulate(global_table, headers="keys", tablefmt="pipe"))
            f.write("\n\n")
            
            # Agregar gráficos
            f.write("## Comparativas Visuales\n\n")
            f.write("### Similitud Semántica por Modelo y Tipo\n\n")
            f.write("![Comparativa de Similitud Semántica](unified_semantic_similarity.png)\n\n")
            
            # Análisis RAG vs NoRAG
            rag_vs_norag = self._analyze_rag_vs_norag()
            f.write("## Análisis RAG vs NoRAG\n\n")
            f.write(rag_vs_norag)
            f.write("\n\n")
            
            # Recomendaciones basadas en datos
            f.write("## Recomendaciones\n\n")
            recommendations = self._generate_recommendations()
            f.write(recommendations)
        
        print(f"✅ Informe unificado generado: {output_file}")
        return output_file
    
    def _create_comparison_chart(self):
        """Crea gráficos comparativos de las evaluaciones."""
        if self.unified_df is None:
            self.load_all_metrics()
        
        # Filtrar solo métricas globales
        global_df = self.unified_df[~self.unified_df.apply(lambda row: 'category' in row, axis=1)].copy()
        
        # Configurar el estilo
        plt.figure(figsize=(12, 8))
        sns.set_style("whitegrid")
        
        # Gráfico de similitud semántica
        ax = sns.barplot(
            data=global_df, 
            x='model', 
            y='semantic_similarity', 
            hue='type',
            palette={"RAG": "blue", "NoRAG": "orange"}
        )
        
        # Personalizar gráfico
        plt.title('Comparativa de Similitud Semántica por Modelo', size=16)
        plt.xlabel('Modelo', size=12)
        plt.ylabel('Similitud Semántica', size=12)
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        # Añadir valores en las barras
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3)
        
        plt.tight_layout()
        chart_path = os.path.join(self.output_dir, 'unified_semantic_similarity.png')
        plt.savefig(chart_path, dpi=300)
        plt.close()
        print(f"✅ Gráfico guardado: {chart_path}")
    
    def export_to_csv(self, output_file="unified_evaluation_data.csv"):
        """Exporta todos los datos unificados a un archivo CSV."""
        if self.unified_df is None:
            self.load_all_metrics()
        
        output_file = os.path.join(self.output_dir, output_file)
        self.unified_df.to_csv(output_file, index=False)
        print(f"✅ Datos exportados a CSV: {output_file}")
        return output_file
    
    def export_to_json(self, output_file="unified_evaluation_data.json"):
        """Exporta todos los datos unificados a un archivo JSON."""
        if self.unified_df is None:
            self.load_all_metrics()
        
        output_file = os.path.join(self.output_dir, output_file)
        self.unified_df.to_json(output_file, orient='records', indent=2)
        print(f"✅ Datos exportados a JSON: {output_file}")
        return output_file

# Ejemplo de uso
if __name__ == "__main__":
    unifier = EvaluationUnifier()
    
    # Cargar todas las métricas
    unifier.load_all_metrics()
    
    # Crear informe unificado
    unifier.create_unified_report()
    
    # Exportar a formatos portables
    unifier.export_to_csv()
    unifier.export_to_json()
    
    print("✅ Proceso completado. Se han generado los siguientes archivos en la carpeta 'unified_results':")
    print("- unified_evaluation_report.md    (Informe completo en Markdown)")
    print("- unified_evaluation_data.csv     (Datos tabulares en CSV)")
    print("- unified_evaluation_data.json    (Datos estructurados en JSON)")
    print("- unified_semantic_similarity.png (Gráfico comparativo)")
