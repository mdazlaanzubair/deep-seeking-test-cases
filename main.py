from tqdm import tqdm
import datetime
import json
from modules.helper import load_data, chunk_data, save_data
from modules.langchain import get_chain, parser

# Load test cases data
test_cases = load_data('data/cleaned_data.json')

# Chunk test cases data
tc_chunks = chunk_data(test_cases, 3)

# Initialize chain for evaluation
models_list = ["deepseek-r1:1.5b", "llama3.2:3b", "mistral:7b"]
active_model = models_list[2]
chain = get_chain(model_name=active_model)

# Lists to hold successful and unsuccessful test cases
success_jobs = []
failed_jobs = []

def evaluate_test_case(test_case, model_name):
    """
    Evaluate a single test case using the specified model.
    """
    start_time = datetime.datetime.now()
    llm_raw_output = ""
    response_dict = {}

    # Remove 'group' from input variables to avoid modifying the original dictionary
    input_variables = {k: v for k, v in test_case.items() if k != 'group'}
    
    try:

        # Invoke the LLM chain
        llm_raw_output = chain.invoke(input_variables)
        end_time = datetime.datetime.now()

        # Parse the raw output
        parsed_response = parser.parse(llm_raw_output)
        response_dict = parsed_response.model_dump()

        # Add metadata to the response
        response_dict.update({
            "time_taken": {
                "start_time": start_time.strftime("%d/%m/%Y %H:%M:%S"),
                "end_time": end_time.strftime("%d/%m/%Y %H:%M:%S"),
                "duration_in_sec": (end_time - start_time).total_seconds(),
            },
            "test_case_id": test_case["test_case_id"],
            "group": test_case["group"],
            "evaluated_by": model_name,
        })

        success_jobs.append(response_dict)
        # print(f"✅ Test case '{test_case['test_case_id']}' of '{test_case['group']}' group evaluated successfully!")

    except Exception as e:
        end_time = datetime.datetime.now()

        # Prepare failure details
        failed_case = {
            "evaluated_by": model_name,
            "test_case": test_case,
            "llm_raw_output": llm_raw_output,
            "error_exception_details": str(e),
            "time_taken": {
                "start_time": start_time.strftime("%d/%m/%Y %H:%M:%S"),
                "end_time": end_time.strftime("%d/%m/%Y %H:%M:%S"),
                "duration_in_sec": (end_time - start_time).total_seconds(),
            },
        }

        failed_jobs.append(failed_case)
        # print(f"❌ Test case '{test_case['test_case_id']}' of '{test_case['group']}' group evaluation failed!")


if chain:
    # Outer progress bar for chunk processing (yellow)
    chunk_progress = tqdm(
        tc_chunks,
        bar_format='[{elapsed}<{remaining}] {n_fmt}/{total_fmt} | {l_bar}{bar} {rate_fmt}{postfix}',
        desc="Processing Chunks",
        colour='yellow',
        position=0  # Position 0 for the outer progress bar
    )

    for chunk_index, chunk in enumerate(chunk_progress, start=1):
        total_test_cases = len(chunk)

        # Update the outer progress bar description
        chunk_progress.set_description(f"Processing Chunk {chunk_index}/{len(tc_chunks)}")

        # Inner progress bar for test case processing (green)
        test_case_progress = tqdm(
            chunk,
            bar_format='[{elapsed}<{remaining}] {n_fmt}/{total_fmt} | {l_bar}{bar} {rate_fmt}{postfix}',
            desc=f"Evaluating Test Cases (Chunk {chunk_index})",
            colour='green',
            position=1  # Position 1 for the inner progress bar
        )

        for i, test_case in enumerate(test_case_progress, start=1):
            # Update the inner progress bar description
            test_case_progress.set_description(f"Evaluating Test Case {i}/{total_test_cases} (Chunk {chunk_index})")
            
            # Perform evaluation
            evaluate_test_case(test_case, active_model)

        # Save results after processing each chunk
        save_data(success_jobs, "data/evaluations/success.json")
        save_data(failed_jobs, "data/evaluations/failed.json")

        # print(f"Chunk {chunk_index}: Success: {len(success_jobs)}/{total_test_cases}, Rejected: {len(failed_jobs)}/{total_test_cases}")

        # Break after processing the first chunk
        break

    print(f"Final Report: Success: {len(success_jobs)}/{total_test_cases}, Rejected: {len(failed_jobs)}/{total_test_cases}")
    
else:
    print("Lang-Chain does not initialized.")
    
    