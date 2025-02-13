from tqdm import tqdm
import datetime
import time
import copy
import json

from modules.helper import load_data, chunk_data, save_data
from modules.langchain import get_chain, parser

# Loading test cases data
test_cases = load_data('data/cleaned_data.json')

# Chunking test cases data
tc_chunks = chunk_data(test_cases, 3)

# Initializing chain for evaluation of test cases
models_list = ["deepseek-r1:1.5b", "llama3.2:1b", "llama3.2", "mistral"]
chain = get_chain(model_name=models_list[1])

# Function to evaluate test_cases using langchain and llm
def evaluate_test_case(test_case, model_name):
    llm_raw_output="" # variable to hold raw llm response
    start_time = "" # variable to hold processing start time
    end_time = "" # variable to hold processing end time
    
    # Dictionary to hold the LLM response
    response_dict = {}
    
    # Extracting input variables value from test_case object
    # Making deep copy to avoid modifying the original dictionary
    input_variables = copy.deepcopy(test_case)

    # Remove 'group' from the copy only
    input_variables.pop('group', None)
    
    # Using try/except to chain invocation process
    try:
        # Updating start time
        start_time = datetime.datetime.now()

        # Hitting LLM for response by passing the input_variables
        llm_raw_output = chain.invoke(input_variables)
        
        # Updating end time
        end_time = datetime.datetime.now()

        # Parsing llm_raw_output using parser
        parsed_response = parser.parse(llm_raw_output)

        # Convert parsed response to dictionary
        response_dict = parsed_response.model_dump()

        # Adding time taken in processing
        response_dict["time_taken"] = {
            # Formatting times as "dd/mm/yyyy hh:mm:ss"
            "start_time": start_time.strftime("%d/%m/%Y %H:%M:%S"),
            "end_time": end_time.strftime("%d/%m/%Y %H:%M:%S"),
            
            # Calculating time taken by LLM for evaluation in seconds
            "duration_in_sec": (end_time - start_time).total_seconds()
        }
        
        
        # Inserting test_case identifiers
        response_dict["test_case_id"] = test_case["test_case_id"],
        response_dict["group"] = test_case["group"]
        response_dict["evaluated_by"] = model_name

        # print(json.dumps(response_dict, indent=2))
        success_jobs.append(response_dict)
        print(f"✅ Test case '{test_case["test_case_id"]}' of '{test_case["group"]}' group evaluated successfully!")
        
    except Exception as e:
        # Preparing failed object for  alter evaluation and debugging
        failed_case = {
            "evaluated_by": model_name,
            "test_case": test_case,
            "llm_raw_output": llm_raw_output,
            "error_exception_details": str(e),
            "time_taken": {
                # Formatting times as "dd/mm/yyyy hh:mm:ss"
                "start_time": start_time.strftime("%d/%m/%Y %H:%M:%S"),
                "end_time": end_time.strftime("%d/%m/%Y %H:%M:%S"),
                
                # Calculating time taken by LLM for evaluation in seconds
                "duration_in_sec": (end_time - start_time).total_seconds()
            }
        }
        
        failed_jobs.append(failed_case)
        
        print(f"❌ Test case '{test_case["test_case_id"]}' of '{test_case["group"]}' group evaluation failed!")


# Lists to hold successful and unsuccessful test cases
success_jobs = []
failed_jobs = []
    
if (chain):
    # Looping through test cases
    for test_case in tqdm(tc_chunks[0], desc="Evaluating Test Cases"):
        # Performing Evaluation
        evaluate_test_case(test_case, models_list[1])
        
        # Saving 
        save_data(success_jobs, "data/evaluations/success.json")
        save_data(failed_jobs, "data/evaluations/failed.json")
        
    print("Success:", len(success_jobs), "/", len(tc_chunks[0]))
    print("Rejected:", len(failed_jobs), "/", len(tc_chunks[0]))
else:
    print("Lang-Chain does not initialized.")



