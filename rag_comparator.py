from norag_evaluator import RAGComparator

comparator = RAGComparator(
    rag_results_dir="evaluation_results/eval_20250901_182451", 
    norag_results_dir="evaluation_results/norag_eval_20250902_145339"
)

comparator.generate_comparison_report("rag_vs_norag_analysis.md")