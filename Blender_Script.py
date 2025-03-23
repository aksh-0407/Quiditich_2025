import bpy
import json

# Load the JSON file
json_path = r"C:\Users\Shivank\Documents\quidich\Motion data\GPT - Origin updated.json"  # Update path
with open(json_path, "r") as file:
    data = json.load(file)

# Reference the Armature
armature_name = "Armature"  # Ensure this matches Blender armature name
armature = bpy.data.objects.get(armature_name)

if not armature:
    print(f"\u26a0\ufe0f Armature '{armature_name}' not found!")
    exit()

# Switch to Pose Mode
bpy.context.view_layer.objects.active = armature
bpy.ops.object.mode_set(mode="POSE")

# Define bone mapping
bone_mapping = {
    "Left_Foot": "Left_Foot",
    "Right_Foot": "Right_Foot",
    "Left_Knee": "Left_Knee",
    "Right_Knee": "Right_Knee",
    "Left_Hip": "Left_Hip",
    "Right_Hip": "Right_Hip",
    "Left_Shoulder": "Left_Shoulder",
    "Right_Shoulder": "Right_Shoulder",
    "Left_Forearm": "Left_Forearm",
    "Right_Forearm": "Right_Forearm",
    "Left_Hand": "Left_Hand",
    "Right_Hand": "Right_Hand",
    "Head": "Head",
    "Neck": "Neck"
}

# List of IK-controlled bones
ik_bones = [
    "Left_Shoulder.001", "Right_Shoulder.001", "Left_Foot.001", "Right_Foot.001",
    "Neck.001", "Left_Arm_IK", "Right_Arm_IK", "Left_Knee_IK", "Right_Knee_IK",
    "Left_Index_IK_Closed", "Left_Middle_IK_Closed", "Left_Ring_IK_Closed",
    "Left_Pinky_IK_Closed", "Left_Thumb_IK_Closed", "Left_Pinky_IK_Open",
    "Left_Ring_IK_Open", "Left_Middle_IK_Open", "Left_Index_IK_Open",
    "Right_Index_IK_Closed", "Right_Middle_IK_Closed", "Right_Ring_IK_Closed",
    "Right_Pinky_IK_Closed", "Right_Thumb_IK_Closed", "Right_Pinky_IK_Open",
    "Right_Ring_IK_Open", "Right_Middle_IK_Open", "Right_Index_IK_Open"
]

# Function to disable/enable IK constraints
def toggle_ik_constraints(enable=False):
    for bone in armature.pose.bones:
        for constraint in bone.constraints:
            if "IK" in constraint.name:
                constraint.mute = not enable  # Disable if enable=False, Enable if enable=True
                print(f"{'✅ Enabled' if enable else '🚫 Disabled'} IK on {bone.name}")

# Disable IK before animation
toggle_ik_constraints(enable=False)

# Loop through frames and set bone positions
for frame_data in data:
    frame_number = frame_data["frameNumber"]
    landmarks = frame_data["landmarks"]

    bpy.context.scene.frame_set(frame_number)  # Set current frame

    for joint_name, bone_name in bone_mapping.items():
        if joint_name in landmarks and bone_name in armature.pose.bones:
            position = landmarks[joint_name]["position"]
            bone = armature.pose.bones[bone_name]

            # Adjust for Blender's coordinate system (Z-Up)
            new_position = (position[0] / 0.07, position[1] / 0.07, position[2] / 0.07)

            # Apply transformation and update
            bone.location = new_position

            # Insert keyframe
            bone.keyframe_insert(data_path="location", frame=frame_number)

            # Debugging: Print if a bone is moving
            print(f"✅ Frame {frame_number}: {bone_name} moved to {new_position}")

# Re-enable IK constraints after animation
toggle_ik_constraints(enable=True)

print("✅ Animation applied successfully!")
