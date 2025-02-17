import json
from modules.stats_helper import get_descriptive_stats, perform_statistical_tests, format_data_to_evaluations, structuring_stats_in_metrics, create_performance_charts
from modules.helper import load_data, save_data


# Load data from processed_results.json
test_cases_scores = load_data("data/evaluations/mixtral-8x7b-32768/archive/processed_results.json")
print(f"✅ Dataset loaded: {len(test_cases_scores)}")

evaluations = format_data_to_evaluations(test_cases_scores)
print(f"✅ Prepared evaluations: {len(evaluations)}")

if __name__ == "__main__":
    print("Descriptive Statistics:")
    print("-----------------------")
    
    results_descriptive = get_descriptive_stats(evaluations)
    print("  ✅ Calculated")
    
    structured_results_descriptive = structuring_stats_in_metrics(results_descriptive)
    print("  ✅ Structured")
    
    save_data(structured_results_descriptive, "data/results/mixtral-8x7b-32768/stats.json")
    print("  ✅ Saved\n")
    
    
    print("Test Statistics:")
    print("----------------")
    
    statistical_tests_results = perform_statistical_tests(evaluations, "quality_score")
    print("  ✅ Calculated")
    
    # Load the JSON data from the 'dataset' string
    structured_statistical_tests_results = json.loads(statistical_tests_results)
    print("  ✅ Converted to JSON")
    
    save_data(structured_statistical_tests_results, "data/results/mixtral-8x7b-32768/adv_stats.json")
    print("  ✅ Saved\n")
    
    
    print("Creating Stats Visuals:")
    print("-----------------------")
    create_performance_charts(structured_results_descriptive, "data/results/mixtral-8x7b-32768/charts")
    print("  ✅ Charts saved\n")
    
    
    