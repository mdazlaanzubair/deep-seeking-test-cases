import pandas as pd
import numpy as np
from scipy.stats import f_oneway, ttest_ind
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
import os


# Custom JSON encoder to handle numpy data types.
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Series):  # Add this check for Pandas Series
            return obj.tolist()  # Convert Series to a list
        return super(NpEncoder, self).default(obj)


# Function to calculates descriptive statistics for each group in the dataset.
def get_descriptive_stats(dataset):
    # Convert dataset to a Pandas DataFrame
    df = pd.DataFrame(dataset)

    # Group by 'group'
    grouped = df.groupby("group")

    # Initialize results dictionary
    results = {}

    # Define metrics for comprehensive analysis
    metrics = [
        "coverage",
        "clarity",
        "edge_and_negative_cases_score",
        "non_functional_coverage",
        "quality_score",
    ]

    # Define weights for weighted score calculation (sum to 1.0)
    weights = {
        "coverage": 0.30,
        "clarity": 0.20,
        "edge_and_negative_cases_score": 0.25,
        "non_functional_coverage": 0.25,
    }

    for group, group_df in grouped:
        total_tc = len(group_df)
        high_quality_cases = (group_df["quality_score"] >= 4).sum()

        # Weighted score calculation using defined weights
        total_weighted_score = sum(
            (group_df[metric] * weight).sum()  # Sum each metric's weighted scores
            for metric, weight in weights.items()
        )

        # Calculate efficiency index (max possible score per case = 5)
        efficiency_index = total_weighted_score / (5 * total_tc)

        # Initialize group results
        group_results = {
            "total_tc": total_tc,
            "high_quality_cases": high_quality_cases,
            "total_weighted_score": total_weighted_score,
            "qtq_ratio": total_weighted_score / total_tc,
            "efficiency_index": efficiency_index,
        }

        # Calculate comprehensive statistics for all metrics
        for metric in metrics:
            group_results[f"avg_{metric}"] = group_df[metric].mean()
            group_results[f"std_{metric}"] = group_df[metric].std()
            group_results[f"median_{metric}"] = group_df[metric].median()
            mode = group_df[metric].mode()
            group_results[f"mode_{metric}"] = mode.iloc[0] if not mode.empty else None
            group_results[f"var_{metric}"] = group_df[metric].var()

        # Add small sample warning
        if total_tc < 30:
            group_results["warning"] = "Small sample size - results may be unreliable"
        results[group] = group_results

    # convert results into valid JSON
    results_json = json.dumps(results, indent=2, cls=NpEncoder)
    return results_json


# Function to performs statistical tests (ANOVA and pairwise t-tests) on the dataset.
def perform_statistical_tests(dataset, test_metric="coverage"):

    # Convert dataset to a Pandas DataFrame
    df = pd.DataFrame(dataset)

    # Group by 'group'
    grouped = df.groupby("group")

    # Store all group data for ANOVA and t-tests
    all_group_data = {}
    for group, group_df in grouped:
        all_group_data[group] = group_df

    # Perform ANOVA test on specified metric
    anova_groups = [data[test_metric] for data in all_group_data.values()]
    anova_result = f_oneway(*anova_groups)

    # Perform pairwise t-tests with multiple comparison correction
    group_names = list(all_group_data.keys())
    t_test_results = {}
    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            group1 = group_names[i]
            group2 = group_names[j]
            t_stat, p_value = ttest_ind(
                all_group_data[group1][test_metric], all_group_data[group2][test_metric]
            )
            key = f"{group1}_vs_{group2}"
            t_test_results[key] = {
                "t_stat": t_stat,
                "p_value": p_value,
                "interpretation": (
                    "Significant" if p_value < 0.05 else "Not significant"
                ),
            }

    # Apply Bonferroni correction
    num_comparisons = len(t_test_results)
    for comparison in t_test_results:
        original_p = t_test_results[comparison]["p_value"]
        corrected_p = min(original_p * num_comparisons, 1.0)
        t_test_results[comparison]["corrected_p_value"] = corrected_p

    # Add statistical results to output
    statistical_tests = {
        "anova": {
            "tested_metric": test_metric,
            "F-statistic": anova_result.statistic,
            "p-value": anova_result.pvalue,
        },
        "pairwise_t_tests": t_test_results,
        "note": "Bonferroni correction applied to pairwise comparisons",
    }

    statistical_tests_json = json.dumps(statistical_tests, indent=2, cls=NpEncoder)
    return statistical_tests_json


# Function to convert the "processed_results" scores data into evaluations format
def format_data_to_evaluations(dataset):
    evaluations = []

    for result in dataset:
        evaluations.append(
            {
                # unique id of a test case
                "test_case_id": result["test_case_id"],
                # score per criteria
                "group": result["group"],
                "coverage": result["evaluation"]["coverage"]["score"],
                "clarity": result["evaluation"]["clarity"]["score"],
                "edge_and_negative_cases_score": result["evaluation"][
                    "edge_and_negative_cases_score"
                ]["score"],
                "non_functional_coverage": result["evaluation"][
                    "non_functional_coverage"
                ]["score"],
                # Calculate the overall weighted or quality score
                "quality_score": (
                    0.3 * result["evaluation"]["coverage"]["score"]
                    + 0.2 * result["evaluation"]["clarity"]["score"]
                    + 0.25
                    * result["evaluation"]["edge_and_negative_cases_score"]["score"]
                    + 0.25 * result["evaluation"]["non_functional_coverage"]["score"]
                ),
            }
        )

    return evaluations


# Function to distribute grouped results into performance metrics
def structuring_stats_in_metrics(dataset):

    structured_stats_results = {
        # Key Performance Metrics
        "Key Performance Metrics": {
            "Total Test Cases": {},
            "High Quality Cases": {},
            "Total Weighted Score": {},
            "Quality-to-Quantity Ratio": {},
            "Efficiency Index": {},
        },
        # Individual Metrics
        "Coverage": {"avg": {}, "std": {}, "median": {}, "mode": {}, "var": {}},
        "Clarity": {"avg": {}, "std": {}, "median": {}, "mode": {}, "var": {}},
        "Edge & Negative Cases Score": {
            "avg": {},
            "std": {},
            "median": {},
            "mode": {},
            "var": {},
        },
        "Non-Functional Coverage": {
            "avg": {},
            "std": {},
            "median": {},
            "mode": {},
            "var": {},
        },
        "Overall Quality Score": {
            "avg": {},
            "std": {},
            "median": {},
            "mode": {},
            "var": {},
        },
    }
    
    # Load the JSON data from the 'dataset' string
    results_data = json.loads(dataset)

    # Now iterate through the loaded dictionary
    for group, stats in results_data.items():
        # grouping all key performance metrics
        structured_stats_results["Key Performance Metrics"]["Total Test Cases"][
            group
        ] = stats["total_tc"]
        structured_stats_results["Key Performance Metrics"]["High Quality Cases"][
            group
        ] = stats["high_quality_cases"]
        structured_stats_results["Key Performance Metrics"]["Total Weighted Score"][
            group
        ] = stats["total_weighted_score"]
        structured_stats_results["Key Performance Metrics"][
            "Quality-to-Quantity Ratio"
        ][group] = stats["qtq_ratio"]
        structured_stats_results["Key Performance Metrics"]["Efficiency Index"][
            group
        ] = stats["efficiency_index"]

        # grouping all individual metrics
        # Coverage
        structured_stats_results["Coverage"]["avg"][group] = stats["avg_coverage"]
        structured_stats_results["Coverage"]["std"][group] = stats["std_coverage"]
        structured_stats_results["Coverage"]["median"][group] = stats["median_coverage"]
        structured_stats_results["Coverage"]["mode"][group] = stats["mode_coverage"]
        structured_stats_results["Coverage"]["var"][group] = stats["var_coverage"]

        # Clarity
        structured_stats_results["Clarity"]["avg"][group] = stats["avg_clarity"]
        structured_stats_results["Clarity"]["std"][group] = stats["std_clarity"]
        structured_stats_results["Clarity"]["median"][group] = stats["median_clarity"]
        structured_stats_results["Clarity"]["mode"][group] = stats["mode_clarity"]
        structured_stats_results["Clarity"]["var"][group] = stats["var_clarity"]

        # Edge & Negative Cases Score
        structured_stats_results["Edge & Negative Cases Score"]["avg"][group] = stats[
            "avg_edge_and_negative_cases_score"
        ]
        structured_stats_results["Edge & Negative Cases Score"]["std"][group] = stats[
            "std_edge_and_negative_cases_score"
        ]
        structured_stats_results["Edge & Negative Cases Score"]["median"][group] = (
            stats["median_edge_and_negative_cases_score"]
        )
        structured_stats_results["Edge & Negative Cases Score"]["mode"][group] = stats[
            "mode_edge_and_negative_cases_score"
        ]
        structured_stats_results["Edge & Negative Cases Score"]["var"][group] = stats[
            "var_edge_and_negative_cases_score"
        ]

        # Non-Functional Coverage
        structured_stats_results["Non-Functional Coverage"]["avg"][group] = stats[
            "avg_non_functional_coverage"
        ]
        structured_stats_results["Non-Functional Coverage"]["std"][group] = stats[
            "std_non_functional_coverage"
        ]
        structured_stats_results["Non-Functional Coverage"]["median"][group] = stats[
            "median_non_functional_coverage"
        ]
        structured_stats_results["Non-Functional Coverage"]["mode"][group] = stats[
            "mode_non_functional_coverage"
        ]
        structured_stats_results["Non-Functional Coverage"]["var"][group] = stats[
            "var_non_functional_coverage"
        ]

        # Overall Quality Score
        structured_stats_results["Overall Quality Score"]["avg"][group] = stats[
            "avg_quality_score"
        ]
        structured_stats_results["Overall Quality Score"]["std"][group] = stats[
            "std_quality_score"
        ]
        structured_stats_results["Overall Quality Score"]["median"][group] = stats[
            "median_quality_score"
        ]
        structured_stats_results["Overall Quality Score"]["mode"][group] = stats[
            "mode_quality_score"
        ]
        structured_stats_results["Overall Quality Score"]["var"][group] = stats[
            "var_quality_score"
        ]

    return structured_stats_results


# Function to create and save visualizations from performance metrics data
def create_performance_charts(data: Dict[str, Any], output_dir: str = "charts") -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    
    # 1. Key Performance Metrics Bar Charts
    metrics = data["Key Performance Metrics"]
    for metric, values in metrics.items():
        plt.figure(figsize=(12, 6))
        bars = plt.bar(values.keys(), values.values())
        plt.xticks(rotation=45, ha='right')
        plt.title(f'{metric} by Model')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{metric.lower().replace(" ", "_")}.png')
        plt.close()
    
    # 2. Coverage Statistics Comparison
    coverage_stats = data["Coverage"]
    metrics = ['avg', 'std', 'median', 'mode', 'var']
    
    plt.figure(figsize=(15, 8))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 3, i)
        values = coverage_stats[metric]
        plt.bar(values.keys(), values.values())
        plt.xticks(rotation=90, fontsize=8)
        plt.title(f'Coverage {metric.upper()}')
        plt.tight_layout()
    plt.savefig(f'{output_dir}/coverage_statistics.png')
    plt.close()
    
    # 3. Clarity Heatmap
    clarity_data = pd.DataFrame(data["Clarity"])
    plt.figure(figsize=(12, 8))
    sns.heatmap(clarity_data, annot=True, cmap='YlOrRd', fmt='.2f')
    plt.title('Clarity Metrics Heatmap')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/clarity_heatmap.png')
    plt.close()
    
    # 4. Edge Cases vs Overall Quality Scatter Plot
    edge_cases = pd.DataFrame(data["Edge & Negative Cases Score"]["avg"], index=[0]).T
    quality = pd.DataFrame(data["Overall Quality Score"]["avg"], index=[0]).T
    
    plt.figure(figsize=(10, 6))
    plt.scatter(edge_cases[0], quality[0])
    for i, model in enumerate(edge_cases.index):
        plt.annotate(model.split('-')[-1], (edge_cases.iloc[i, 0], quality.iloc[i, 0]))
    plt.xlabel('Edge Cases Score')
    plt.ylabel('Overall Quality Score')
    plt.title('Edge Cases vs Overall Quality')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/edge_cases_vs_quality.png')
    plt.close()
    
    # 5. Non-Functional Coverage Radar Chart
    nf_coverage = data["Non-Functional Coverage"]["avg"]
    models = list(nf_coverage.keys())
    values = list(nf_coverage.values())
    
    angles = np.linspace(0, 2*np.pi, len(models), endpoint=False)
    values = np.concatenate((values, [values[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.split('-')[-1] for m in models], size=8)
    plt.title('Non-Functional Coverage by Model')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/nonfunctional_coverage_radar.png')
    plt.close()

