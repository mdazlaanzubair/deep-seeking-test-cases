# QA Test Eval

This repository related to research conduct over comparitive study on writing **Manual Test Cases** using LLMs (GPT 4o and o1). The purpose of research is to answer following questions:

> 1. Does AI models are better in writing quality test cases than Human Engineers?
> 2. Does the reasoning capabilities of an LLMs really impact the quality of test cases?
> 3. Can we achive the similar **achive the quality of reasoning LLMs by just using prompt engineering** on a regular LLM?

Following are the libraries that this Python project require to evaluate the quality of test cases written by `Human Engineers`, `GPT 4o` and `o1`:

- [LangChain](https://github.com/hwchase17/langchain)
- [langchain-ollama](https://github.com/ollama/ollama)
- [tqdm](https://github.com/tqdm/tqdm)
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)

The project includes an script (`main.py`) that:
- Interacts with an LLM via LangChain using `langchain-ollama`.
- Evaluate and score the manual software quality test cases using LLM running locally using Ollama.
- Perform statistics on the evaluated score using `pandas` and `scipy` to generate insights from data.
- Generates and visualizes those statiscs using `matplotlib` and `seaborn`.
- Displays progress using `tqdm`.

## Prerequisites

- **Python 3.x**: [Download Python](https://www.python.org/downloads/)
- **Git**: [Download Git](https://git-scm.com/downloads)
- **VS Code** (Optional but recommended): [Download VS Code](https://code.visualstudio.com/)

## Getting Started

Follow these steps to clone the repository, set up your virtual environment, install dependencies, and run the project locally.

### 1. Clone the Repository

Open your terminal and run:

```bash
git clone https://github.com/yourusername/langchain_project.git
```
```bash
cd langchain_project
```

### 2. Create a Virtual Environment
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

### 3. Install Dependencies
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
pip install langchain langchain-ollama tqdm pandas numpy matplotlib seaborn
```
- **On macOS/Linux:**
```bash
pip3 install langchain langchain-ollama tqdm pandas numpy matplotlib seaborn
```

### 4. Run the Project
Run the main script with:
```bash
python main.py
```
You should see the following:
- A printed preview of a DataFrame generated from random data.
- A progress bar (powered by tqdm) simulating work.
- A plot window displaying the generated data (using matplotlib and seaborn).
- The output of an LLM query printed in the terminal.

### Troubleshooting

- **Virtual Environment:**
Ensure that your virtual environment is activated. You should see (env) in your terminal prompt.

- **Dependency Installation:**
Verify that you have a stable internet connection and the necessary permissions to install packages.

- **LangChain / Ollama Setup:**
Make sure that your Ollama service or local model is correctly configured and running. Refer to the [**langchain-ollama documentation**](https://python.langchain.com/api_reference/community/llms/langchain_community.llms.ollama.Ollama.html "langchain-ollama documentation") for additional guidance.