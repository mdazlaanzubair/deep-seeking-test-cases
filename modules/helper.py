import json
# import datetime
from langchain.schema import AIMessage
import time

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, AIMessage):
            return {
                "content": obj.content,
                "additional_kwargs": obj.additional_kwargs,
                "type": "AIMessage"
            }
        return super().default(obj)

# Loads data from a JSON file
def load_data(json_file):
    with open(json_file) as f:
        data = json.load(f)
    return data

# Chunk large data into smaller pieces
def chunk_data(data, chunk_size):
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

# In order to avoid losing status of successfully evaluated cases and 
# unsuccessful cases due to any crash (i.e. code, server, internet, etc...)
# This function is to save each iteration into their respective JSON files 
# based on successfully and unsuccessfully evaluated state
def save_data(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4, cls=CustomEncoder)


# Return a dictionary with formatted start, end times and duration
def format_time_info(start, end):
    return {
        "start_time": start.strftime("%d/%m/%Y %H:%M:%S"),
        "end_time": end.strftime("%d/%m/%Y %H:%M:%S"),
        "duration_in_sec": (end - start).total_seconds(),
    }

# Function to filter out already processed test cases
def filter_unprocessed_test_cases(all_test_cases, processed_test_cases):
    # Create a set of (test_case_id, group) tuples for quick lookup
    processed_set = {(case["test_case_id"], case["group"]) for case in processed_test_cases}

    # Include only those test cases whose (test_case_id, group) is not in the processed_set
    return [
        case for case in all_test_cases
        if (case["test_case_id"], case["group"]) not in processed_set
    ]

# Function to count no of token utilized
def calculate_tokens(test_cases):
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0

    for case in test_cases:
        usage_metadata = case.get("usage_metadata", {})
        total_input_tokens += usage_metadata.get("input_tokens", 0)
        total_output_tokens += usage_metadata.get("output_tokens", 0)
        total_tokens += usage_metadata.get("total_tokens", 0)

    print(f"total_input_tokens: {total_input_tokens}")
    print(f"total_output_tokens: {total_output_tokens}")
    print(f"total_tokens: {total_tokens}")

# Function to print rate limit info
def rate_limit_logger(condition: str, active_api_key: int, request_made: int, token_made: int):
    line = "-" * 30
    print(line)
    print(f"Under {condition} Condition Block:")
    print(line)
    print(f"\nPer {condition} Limit Exceeded:")
    print("-" * 28)
    print("Active API Key:", active_api_key)
    print(f"RP{condition[0]}:", request_made)  # RPD or RPM
    print(f"TP{condition[0]}:", token_made)  # TPD or TPM
    print(f"Switching API Key - Resetting {condition} Counters")
    print("-" * 28, "\n")

# Delay to wait (in seconds) when rate limit is exceeded
HALF_MINUTE_DELAY = 30
MINUTE_DELAY = 60
HALF_DAY_DELAY = 12 * 3600
DAY_DELAY = 24 * 3600

# Function to delay as per conditions
def wait_for_reset(condition):
    if condition == 'minute':
        print(f"Minute limit reached. Waiting {MINUTE_DELAY} seconds...")
        time.sleep(MINUTE_DELAY)
        print(f"Wait is over, script started again...")
    elif condition == 'half-minute':
        print(f"Minute limit reached. Waiting {HALF_MINUTE_DELAY} seconds...")
        time.sleep(HALF_MINUTE_DELAY)
        print(f"Wait is over, script started again...")
    elif condition == 'day':
        print(f"Daily limit reached. Waiting {DAY_DELAY} seconds...")
        time.sleep(DAY_DELAY)
        print(f"Wait is over, script started again...")
    elif condition == 'half-day':
        print(f"Daily limit reached. Waiting {HALF_DAY_DELAY} seconds...")
        time.sleep(HALF_DAY_DELAY)
        print(f"Wait is over, script started again...")



