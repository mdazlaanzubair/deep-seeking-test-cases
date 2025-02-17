import time
import sys
import os
import datetime

from modules.langchain_helper import get_groq_chain, parser
from modules.helper import load_data, chunk_data, save_data, rate_limit_logger, format_time_info, filter_unprocessed_test_cases, wait_for_reset, calculate_tokens

# Importing list of API Keys in order to increase the
# Rate Limit Per Minute and Day
from modules.api_keys import api_keys

if len(api_keys) <= 0:
    print("No API Key Found. Load API Keys list")
    exit()


# Load test cases data
# 1. Load all test cases
# 2. Load all processed cases
# 3. Removing processed cases from all test cases and start the evaluation script
all_test_cases = load_data("data/cleaned_data.json")
processed_test_cases = load_data(
    "data/evaluations/mixtral-8x7b-32768/archive/processed_results.json"
)
test_cases = filter_unprocessed_test_cases(all_test_cases, processed_test_cases)
calculate_tokens(processed_test_cases)
print(f"\n\nTotal Cases: {len(all_test_cases)}")
print(f"Processed Cases: {len(processed_test_cases)}")
print(f"Remaining Cases: {len(test_cases)}\n\n")

if len(test_cases) <= 0:
    print(
        "No test case to process. Kindly check if the test cases are all processed or not loaded."
    )
    exit()


# Initialize chain for evaluation
models_list = ["llama3-70b-8192", "mixtral-8x7b-32768", "qwen-2.5-32b"]
active_model = models_list[1]
active_api_key = 0

# Rate limit constants
REQUEST_PER_MINUTE = 30
REQUEST_PER_DAY = 14400
TOKENS_PER_MINUTE = 5000
TOKENS_PER_DAY = 500000
DELAY = 500000

# Initializing tracking variables to handle rate limiting error
request_made_per_minute = 0
request_made_per_day = 0
token_made_per_minute = 0
token_made_per_day = 0

# Lists to hold successful, unsuccessful and remaining test cases
# so that we can have the last state if the program crashes
success_jobs = []
failed_jobs = []
remaining_jobs = []


# -------------------
# 3. GROQ CHAIN MAKER
# -------------------
# Evaluate a single test case using the specified model.
def evaluate_test_case(test_case, model_name, api_key):
    global request_made_per_minute, request_made_per_day
    global token_made_per_minute, token_made_per_day
    global success_jobs, failed_jobs

    print("   [DEBUG LOG 1] - Preparing Chain")

    chain = get_groq_chain(model_name, api_key)
    start_time = datetime.datetime.now()
    llm_raw_output = ""

    # Prepare input without modifying original test_case
    input_variables = {k: v for k, v in test_case.items() if k != "group"}

    print("   [DEBUG LOG 2] - Preparing Input Variables")

    try:
        print("   [DEBUG LOG 3] - Invoking Chain")

        llm_raw_output = chain.invoke(input_variables)

        print("   [DEBUG LOG 4] - LLM Response Success")

        end_time = datetime.datetime.now()
        content = llm_raw_output.content
        response_metadata = llm_raw_output.response_metadata
        usage_metadata = llm_raw_output.usage_metadata
        parsed_response = parser.parse(content)

        print("   [DEBUG LOG 5] - LLM Response Parsed Success")

        success_case = parsed_response.model_dump()

        # Add metadata
        success_case.update(
            {
                "evaluated_by": model_name,
                "time_taken": format_time_info(start_time, end_time),
                "group": test_case["group"],
                "test_case_id": test_case["test_case_id"],
                "response_metadata": response_metadata,
                "usage_metadata": usage_metadata,
            }
        )

        # Update rate limit counters based on usage tokens
        tokens_used = usage_metadata["total_tokens"]
        request_made_per_minute += 1
        request_made_per_day += 1
        token_made_per_minute += tokens_used
        token_made_per_day += tokens_used

        success_jobs.append(success_case)
    except Exception as e:
        print("   [DEBUG LOG 6] - Exception Occur")

        if "invalid_api_key" in str(e):
            print(
                "   [ERROR] - Invalid API Key. Please check your API key configuration."
            )
            print("   Test Case ID:", test_case["test_case_id"])
            print("   API Key:", api_key)
            sys.exit(1)  # Use sys.exit() with an exit code

        end_time = datetime.datetime.now()
        failed_case = {
            "evaluated_by": model_name,
            "test_case": test_case,
            "llm_raw_output": llm_raw_output,
            "error_exception_details": str(e),
            "time_taken": format_time_info(start_time, end_time),
        }
        failed_jobs.append(failed_case)

    print("   [DEBUG LOG 7] - Coming Out from evaluation function")


# ===========================================
# MAIN EXECUTION
# ===========================================
total_cases = len(test_cases)

# Looping through test cases
if total_cases <= 0:

    for i, test_case in enumerate(test_cases):
        print(f"\nEvaluation of Case No. {i} - {test_case['test_case_id']} - Started")

        # Perform evaluation under rate limita
        if (
            request_made_per_minute >= REQUEST_PER_MINUTE
            or token_made_per_minute >= TOKENS_PER_MINUTE
        ):
            # Logging rate limit info
            rate_limit_logger(
                "Minute", active_api_key, request_made_per_minute, token_made_per_minute
            )

            if active_api_key < len(api_keys) - 1:
                # Rotate to next API key and reset counters
                active_api_key += 1
            else:
                # Resetting active_api_key
                active_api_key = 0

            # After waiting, reset counters for the current key.
            request_made_per_minute = 0
            token_made_per_minute = 0

            # All keys are exhausted; decide whether to wait for minute or day reset.
            # Here, as a simple approach, we wait for a minute.
            # wait_for_reset('half-minute')

        elif (
            request_made_per_day >= REQUEST_PER_DAY
            or token_made_per_day >= TOKENS_PER_DAY
        ) and (active_api_key < len(api_keys) - 1):
            # Logging rate limit info
            rate_limit_logger(
                "Minute", active_api_key, request_made_per_day, token_made_per_day
            )

            if active_api_key < len(api_keys) - 1:
                # Rotate to next API key and reset counters
                active_api_key += 1
            else:
                # Resetting active_api_key
                active_api_key = 0

            # After waiting, reset counters for the current key.
            request_made_per_day = 0
            token_made_per_day = 0

            # All keys are exhausted; decide whether to wait for minute or day reset.
            # Here, as a simple approach, we wait for a minute.
            # wait_for_reset('half-minute')

        # Process the current test case with the active API key
        current_api_key = api_keys[active_api_key]
        evaluate_test_case(test_case, active_model, current_api_key)

        print(f"Evaluation of Case No. {i} - {test_case['test_case_id']} - Completed")

        # Adding 1 sec delay between each request to avoid spamming
        time.sleep(1)

        # Get remaining test cases (excluding the current one)
        remaining_jobs = test_cases[i + 1 :]

        # Save results after processing every 100 test case
        # So, that we have the final picture of the cases when the program crash or ends
        if i % 100 == 0:
            save_data(success_jobs, "data/evaluations/mixtral-8x7b-32768/success.json")
            save_data(failed_jobs, "data/evaluations/mixtral-8x7b-32768/failed.json")
            save_data(
                remaining_jobs, "data/evaluations/mixtral-8x7b-32768/remaining.json"
            )

        # Break after processing the first chunk
        # if i < 2:
        #     break

    print("\n------------")
    print("Final Report")
    print("------------")
    print(f"- Remaining: {len(remaining_jobs)} / {total_cases}")
    print(f"- Success: {len(success_jobs)} / {total_cases}")
    print(f"- Failed: {len(failed_jobs)} / {total_cases}")
