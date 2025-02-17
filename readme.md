# Comparative Study of LLMs for Manual Test Case Evaluation

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains a Python script designed to conduct research on the comparative effectiveness of Large **Language Models (LLMs)** in evaluating manual test cases. The research aims to explore whether AI models can match or exceed human engineers in assessing the quality of test cases, and to investigate the impact of LLM reasoning capabilities and prompt engineering on evaluation quality.

This project uses **Langchain** and **Ollama** to evaluate test cases against predefined quality criteria, providing a structured, data-driven approach to test case assessment.

### Research Questions

This research endeavors to answer the following key questions:

1.  **Quality Comparison:** Are AI models capable of writing higher quality test cases compared to human software engineers?
2.  **Reasoning Impact:** Does the inherent reasoning capability of different LLMs significantly influence the quality of test case evaluations they produce?
3.  **Prompt Engineering vs. Model Reasoning:** Can prompt engineering applied to a standard LLM achieve comparable reasoning quality to a more advanced LLM, specifically in the context of test case evaluation?

## Code Structure

The codebase is structured as follows:

*   **`data/`**: This directory contains data files:

    *   **`raw_data.json`**:  Raw data of test cases imported from google sheets.

    *   **`cleaned_data.json`**:  Cleaned and structured copy of raw data containing the manual test cases to be evaluated.

    *   **`data/evaluations/[model_name]`**:  Directory to store evaluation results <br /> <br /> 
       `NOTE: Folder and file structure will be same, just group into the processing model's name folder for better arrangements`:

        *   `archive/processed_results.json`:  Archive folder contain processed test cases evaluation to save them from separately from other files, basically its a copy of `success.json`. 

        *   `remaining.json`:  JSON file storing remaining test case that are left to be processed.

        *   `success.json`:  JSON file storing successful test case evaluations.

        *   `failed.json`: JSON file storing details of failed test case evaluations.

    *   **`data/results/[model_name]`**:  Directory to store evaluation results <br /> <br /> 
       `NOTE: Folder and file structure will be same, just group into the processing model's name folder for better arrangements`:

        *   `stats.json`:  JSON file storing `group-wise` calculated stats over evaluation score of each test case.

        *   `adv_stats.json`:  JSON file storing group-wise relation between the scoring performing `ANOVA` and `T-Test`.


*   **`modules/`**:  This directory houses Python modules:

    *   **`helper.py`**: Contains helper functions for data loading, chunking, and saving.
    
    *   **`stats_helper.py`**: This fil contains some helper functions related to calculate stats of the evaluated results:

        *   It has function that calculates descriptive statistics for each group in the dataset.

        *   A function that performs statistical tests (ANOVA and pairwise t-tests) on the dataset.
        
        *   And a function that calculate comprehensive statistics and comparative metrics for test case groups.

    *   **`langchain_helper.py`**:  Sets up the Langchain components, including:

        *   Pydantic model definitions for structured output parsing.

        *   Prompt template for guiding LLM evaluation.
        
        *   Chain creation using Langchain and Ollama.

*   **`main.py`**: The main script to run the test case evaluation process.
*   **`groc_main.py`**: Same script as `main.py`. It just uses **Groq API** for test case evaluation process. <br /> <br />
    `NOTE: Kindly ensure to import your API Keys when using Groq.`
    ```bash
        # Step 1 - Rename the `example_api_key.pys` file to `api_keys.py`
        # Step 2 - Paste all your API Keys into it
    ```
*   **`calc_stats.py`**: Script to calculate stats of each group based on the evaluation scores provided by any of the above script i.e. `main.py` or `groc_main.py`. <br /> <br />
    `NOTE: Kindly ensure to evaluate all test cases before running this stats script.`

## Setup and Usage

To run this script, you'll need to set up your environment with the necessary dependencies and models. 

Following are the libraries that this Python project require to evaluate the quality of test cases:

- [**LangChain**](https://github.com/hwchase17/langchain)
- [**langchain-ollama**](https://github.com/ollama/ollama)
- [**langchain-groq**](https://python.langchain.com/v0.1/docs/integrations/chat/groq/)
- [**tqdm**](https://github.com/tqdm/tqdm)
- [**pandas**](https://pandas.pydata.org/)
- [**numpy**](https://numpy.org/)
- [**matplotlib**](https://matplotlib.org/)
- [**seaborn**](https://seaborn.pydata.org/)


### Prerequisites

- **Python 3.x**: [**Download Python**](https://www.python.org/downloads/)
- **Git**: [**Download Git**](https://git-scm.com/downloads)
- **VS Code** (Optional but recommended): [**Download VS Code**](https://code.visualstudio.com/)
- **Ollama**: Ensure you have Ollama installed and running. Ollama is used to host and serve the LLMs locally. You can download it from [**https://ollama.com/**](https://ollama.com/).
- **LLMs**: Pull the required LLMs using Ollama. The script is configured to use `llama3.2:3b`, or `mistral:7b`. You can pull these models using Ollama CLI:

    ```bash
    ollama pull llama3.2:3b
    ```
    ```bash
    ollama pull mistral:7b
    ```
- **langchain-groq**: In case you wanted to use `Groq API`, refer this [**official documentation**](https://console.groq.com/docs/quickstart) from [**groqCloud**](https://groq.com/).

### Getting Started

Follow these steps to clone the repository, set up your virtual environment, install dependencies, and run the project locally.

#### 1. Clone the Repository

Open your terminal and run:

```bash
git clone https://github.com/mdazlaanzubair/deep-seeking-test-cases.git
```
```bash
cd deep-seeking-test-cases
```

#### 2. Create a Virtual Environment
It is recommended to use a virtual environment to manage dependencies.
- **On Windows:**
```bash
python -m venv env
```
 - **Activate the virtual environment:**
```bash
.\env\Scripts\activate
```
- **On macOS/Linux:**
```bash
python3 -m venv env
```
 - **Activate the virtual environment:**
```bash
source env/bin/activate 
```

#### 3. Install Dependencies
If a requirements.txt file is available, install the dependencies by running:
- **On Windows:**
```bash
pip install -r requirements.txt
```
- **On macOS/Linux:**
```bash
pip3 install -r requirements.txt
```
Alternatively, you can install the necessary packages manually:
- **On Windows:**
```bash
pip install langchain langchain-ollama langchain-groq tqdm pandas numpy matplotlib seaborn scipy
```
- **On macOS/Linux:**
```bash
pip3 install langchain langchain-ollama langchain-groq tqdm pandas numpy matplotlib seaborn scipy
```

#### 4. Run the Project
1.  **Prepare Test Case Data:** Ensure your test case data is in the `data/cleaned_data.json` file and conforms to the expected structure (as implied by the script's input variables).

2. **Run the main script** with:
    - **On Windows:**
    ```bash
    python main.py
    ```
    - **On macOS/Linux:**
    ```bash
    python3 main.py
    ```
3.  **Monitor Progress:** The script uses `tqdm` to display progress bars for chunk and test case processing in the console.

4.  **View Results:** After execution, the evaluation results will be saved in the `data/evaluations/` directory:
    *   `success.json`: Contains detailed JSON outputs for each successfully evaluated test case, including scores for coverage, clarity, edge cases, non-functional coverage, and justifications.

    *   `failed.json`: Contains details of any test cases that failed during evaluation, including error messages and raw LLM output (if available).

#### Configuration

*   **Model Selection:**  The `main.py` script uses `mistral:7b` as the active model by default (`active_model = models_list[2]`). To change the model, modify the `models_list` and the index for `active_model` in the `main.py` script. Ensure the model you select is pulled via Ollama.

*   **Chunk Size:** The `chunk_data` function in `modules/helper.py` is set to chunk test cases into groups of 3 (`chunk_data(test_cases, 3)`).  Adjust the chunk size as needed for performance or batch processing preferences.

*   **Prompt Template:** The prompt template used for evaluation is defined in `modules/langchain.py` within the `get_prompt()` function. You can customize this prompt to adjust the evaluation criteria, instructions, or scoring scale.

### Data Format

The script expects the input data in `data/cleaned_data.json` to be a list of dictionaries, where each dictionary represents a test case and includes the following keys (input variables for the prompt):

*   `software_name`
*   `software_desc`
*   `test_case_id`
*   `test_module`
*   `test_feature`
*   `test_case_title`
*   `test_case_description`
*   `pre_conditions`
*   `test_steps`
*   `test_data`
*   `expected_outcome`
*   `severity_status`
*   `group` *(This field is used for grouping and is removed before passing to the LLM in order to remove biasness)*

The output is structured in **JSON format** as defined by the **Pydantic models** in `modules/langchain.py`, providing scores and reasons for each evaluation criterion.


## Troubleshooting

- **Virtual Environment:**
Ensure that your virtual environment is activated. You should see (venv) in your terminal prompt.

- **Dependency Installation:**
Verify that you have a stable internet connection and the necessary permissions to install packages.

- **LangChain / Ollama Setup:**
Make sure that your Ollama service or local model is correctly configured and running. Refer to the [**langchain-ollama documentation**](https://python.langchain.com/api_reference/community/llms/langchain_community.llms.ollama.Ollama.html "langchain-ollama documentation") for additional guidance.

## License

This project is licensed under the **MIT License** - see the [**LICENSE**](https://opensource.org/license/MIT) file for details.
