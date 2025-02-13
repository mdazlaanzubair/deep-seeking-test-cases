import json

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
def save_data(data, json_file):
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)
    return json_file


