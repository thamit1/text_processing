import json

def convert_to_json(file_path, output_path):
    with open(file_path, 'r') as file:
        # Read the input text from the file
        input_text = file.read()
    
    # Split the input text into lines
    lines = input_text.strip().split('\n')
    
    # Initialize an empty list to store the JSON data
    json_data = []
    
    for line in lines:
        # Split each line into key and description parts
        parts = line.split(':', 1)
        if len(parts) == 2:
            key, description = parts
            # Create a dictionary for each entry
            entry = {
                "description": description.strip(),
                "summary": description.strip()  # Summary is the same as description
            }
            json_data.append(entry)
    
    # Convert the list of dictionaries to JSON format
    with open(output_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

# File paths
input_file_path = 'samples/requirements_nfr.txt'
output_file_path = 'ticket_summaries.json'

# Convert the input text from the file to JSON and write to output file
convert_to_json(input_file_path, output_file_path)
print(f"JSON output written to {output_file_path}")
