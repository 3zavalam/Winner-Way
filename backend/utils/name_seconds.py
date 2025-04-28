import re 

# Function to clean names and replace special characters
def clean_name(name):
    """Clean the name by removing invalid characters and replacing spaces with underscores."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', name.strip()).replace(" ", "_")

# Function to convert HH:MM:SS time format to seconds
def time_to_seconds(time_str):
    """Converts time in HH:MM:SS format to seconds."""
    parts = time_str.split(":")
    if len(parts) == 3:  # HH:MM:SS
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    elif len(parts) == 2:  # MM:SS
        return int(parts[0]) * 60 + int(parts[1])
    return 0