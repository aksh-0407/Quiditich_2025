
# Load the JSON file 
json_file_path = "/mnt/data/GPT edited final pose data.json"
with open(json_file_path, "r") as file:
    data = json.load(file)

# Extract the Left_Foot position from frame 1 to use as the new origin
origin = data[0]["landmarks"]["Left_Foot"]["position"]

# Adjust all positions so that Left_Foot in frame 1 is at (0,0,0)
for frame in data:
    for joint in frame["landmarks"]:
        position = frame["landmarks"][joint]["position"]
        frame["landmarks"][joint]["position"] = [
            position[0] - origin[0],
            position[1] - origin[1],
            position[2] - origin[2],
        ]

# Save the updated JSON file
updated_json_path = "/mnt/data/normalized_pose_data.json"
with open(updated_json_path, "w") as file:
    json.dump(data, file, indent=4)

# Return the updated file path
updated_json_path
