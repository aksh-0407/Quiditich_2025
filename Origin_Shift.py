





import json

# Load the original JSON file
json_file_path = "/mnt/data/initial pose data.json"
with open(json_file_path, "r") as file:
    data = json.load(file)

# Define the mapping of JSON reference points to Blender bone names
bone_mapping = {
    "LEFT_ANKLE": "Left_Foot",
    "RIGHT_ANKLE": "Right_Foot",
    "LEFT_KNEE": "Left_Knee",
    "RIGHT_KNEE": "Right_Knee",
    "LEFT_HIP": "Left_Hip",
    "RIGHT_HIP": "Right_Hip",
    "LEFT_SHOULDER": "Left_Shoulder",
    "RIGHT_SHOULDER": "Right_Shoulder",
    "LEFT_ELBOW": "Left_Forearm",
    "RIGHT_ELBOW": "Right_Forearm",
    "LEFT_WRIST": "Left_Hand",
    "RIGHT_WRIST": "Right_Hand"
}

# Identify bones with IK constraints based on naming conventions
ik_bones = [bone for bone in [
    "Left_Arm_IK", "Right_Arm_IK",
    "Left_Leg_IK", "Right_Leg_IK",
    "Left_Hand_IK", "Right_Hand_IK",
    "Left_Knee_IK", "Right_Knee_IK",
    "Head_IK"
] if bone in bone_mapping.values()]

# Process the JSON: rename reference points and remove unneeded ones
for frame in data:
    new_landmarks = {}
    for ref_point, values in frame["landmarks"].items():
        if ref_point in bone_mapping:
            new_landmarks[bone_mapping[ref_point]] = values
    frame["landmarks"] = new_landmarks

# Save the updated JSON file
updated_json_path = "/mnt/data/updated_pose_data.json"
with open(updated_json_path, "w") as file:
    json.dump(data, file, indent=4)

# Return the updated file path and list of IK bones
updated_json_path, ik_bones




