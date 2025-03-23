Combine these three scripts into one

# After processing all frames, save the 3D points to a JSON file
import json

def save_pose_data_for_unity(all_3d_points):
    unity_data = []
    
    # MediaPipe pose landmark names for reference in Unity
    landmark_names = [name for name in dir(mp_pose.PoseLandmark) if not name.startswith('_')]
    
    for frame_idx, points in enumerate(all_3d_points):
        frame_data = {
            "frameNumber": frame_idx + 350,  # Adjust based on your starting frame
            "landmarks": {}
        }
        
        for i, point in enumerate(points):
            if i < len(landmark_names):
                # Convert to Unity's coordinate system (Y-up, right-handed)
                # MediaPipe uses Y-down, so we need to flip Y
                unity_x = point[0]
                unity_y = point[2]  # In Unity, Y is up (MediaPipe's Z)
                unity_z = point[1]  # Swap Y and Z for Unity
                
                frame_data["landmarks"][landmark_names[i]] = {
                    "position": [float(unity_x), float(unity_y), float(unity_z)]
                }
        
        unity_data.append(frame_data)
    
    # Save to file
    with open('/content/drive/MyDrive/Quidich-HACKATHON-25/pose_data_for_unity.json', 'w') as f:
        json.dump(unity_data, f, indent=2)
    
    print("Saved pose data for Unity integration")

# Call this function after processing all frames
save_pose_data_for_unity(all_3d_points)
