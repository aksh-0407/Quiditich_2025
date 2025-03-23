import cv2
import mediapipe as mp
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import time
import sys
from scipy.spatial.distance import cdist

"""
Cricket 3D Pose Processing Script
--------------------------------
This script processes cricket player video frames to:
1. Extract pose landmarks for multiple players using MediaPipe
2. Triangulate 3D points from two camera views for each player
3. Generate 3D skeleton visualizations (static and animated) for each player
4. Create Unity-compatible JSON files with the 3D pose data for each player
"""

# Paths
# Paths
c233_path = r'C:\Users\aksh0\Desktop\Hackenza\Quidich-HACKATHON-25\233_im'
c235_path = r'C:\Users\aksh0\Desktop\Hackenza\Quidich-HACKATHON-25\235_im'
intrinsic_path = r'C:\Users\aksh0\Desktop\Hackenza\Quidich-HACKATHON-25\intrinsic.json'
extrinsic_path = r'C:\Users\aksh0\Desktop\Hackenza\Quidich-HACKATHON-25\extrinsic.json'
output_dir = 'output'  # Directory for all outputs
unity_dir = 'unity_data'  # Directory for Unity files

# Create output directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(unity_dir, exist_ok=True)

print("Loading camera calibration data...")
# Load camera calibration data
try:
    with open(intrinsic_path, 'r') as f:
        intrinsic_data = json.load(f)

    with open(extrinsic_path, 'r') as f:
        extrinsic_data = json.load(f)
        
    # Extract camera matrices
    K1 = np.array(intrinsic_data['C233']['camera_matrix'])
    D1 = np.array(intrinsic_data['C233']['distortion_coefficients'])
    K2 = np.array(intrinsic_data['C235']['camera_matrix'])
    D2 = np.array(intrinsic_data['C235']['distortion_coefficients'])

    # Extract rotation and translation matrices
    R1 = cv2.Rodrigues(np.array(extrinsic_data['rotation_vectors']['C233']))[0]
    T1 = np.array(extrinsic_data['translation_vectors']['C233']).reshape(3, 1)
    R2 = cv2.Rodrigues(np.array(extrinsic_data['rotation_vectors']['C235']))[0]
    T2 = np.array(extrinsic_data['translation_vectors']['C235']).reshape(3, 1)

    # Compute projection matrices
    P1 = K1 @ np.hstack((R1, T1))
    P2 = K2 @ np.hstack((R2, T2))
    
    print("Camera calibration data loaded successfully")
except Exception as e:
    print(f"Error loading camera calibration data: {e}")
    sys.exit(1)

print("Setting up MediaPipe pose detector...")
# MediaPipe pose detector with multi-person detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,  # Increased complexity for better multi-person detection
    enable_segmentation=False,
    min_detection_confidence=0.1,  # Lower threshold to detect more poses
    min_tracking_confidence=0.3
)
mp_drawing = mp.solutions.drawing_utils

# Skeleton connections for 3D visualization
SKELETON_CONNECTIONS = [
    # Face
    (0, 1), (0, 2), (1, 3), (2, 4),  # Nose to eyes and eyes to ears
    
    # Torso
    (11, 12), (12, 24), (24, 23), (23, 11),  # Hips and shoulders
    
    # Arms
    (11, 13), (13, 15), (15, 17), (17, 19), (19, 15),  # Left arm
    (12, 14), (14, 16), (16, 18), (18, 20), (20, 16),  # Right arm
    
    # Legs
    (23, 25), (25, 27), (27, 29), (29, 31), (31, 27),  # Left leg
    (24, 26), (26, 28), (28, 30), (30, 32), (32, 28),  # Right leg
    
    # Shoulders to hips
    (11, 23), (12, 24),
    
    # Neck and face
    (11, 0), (12, 0)
]

def triangulate_point(P1, P2, point1, point2):
    """Triangulate a 3D point from two 2D points and projection matrices"""
    A = np.array([
        point1[0] * P1[2] - P1[0],
        point1[1] * P1[2] - P1[1],
        point2[0] * P2[2] - P2[0],
        point2[1] * P2[2] - P2[1]
    ])
    _, _, V = np.linalg.svd(A)
    X = V[-1]
    return X[:3] / X[3]

def get_pose_center(pose_landmarks, frame_shape):
    """Calculate the center point of a pose"""
    if not pose_landmarks:
        return None
    
    # Use hip center (midpoint between left and right hip)
    left_hip = np.array([
        pose_landmarks.landmark[23].x * frame_shape[1],
        pose_landmarks.landmark[23].y * frame_shape[0]
    ])
    right_hip = np.array([
        pose_landmarks.landmark[24].x * frame_shape[1],
        pose_landmarks.landmark[24].y * frame_shape[0]
    ])
    
    return (left_hip + right_hip) / 2

def match_players(prev_centers, curr_centers, max_distance=100):
    """Match players between frames based on their center positions"""
    if not prev_centers or not curr_centers:
        return list(range(len(curr_centers)))
    
    # Calculate pairwise distances between previous and current centers
    distances = cdist(prev_centers, curr_centers)
    
    # Initialize assignments
    assignments = [-1] * len(curr_centers)
    used_prev = set()
    used_curr = set()
    
    # First pass: assign closest matches within max_distance
    for i in range(len(prev_centers)):
        if i in used_prev:
            continue
            
        min_dist_idx = np.argmin(distances[i])
        if distances[i][min_dist_idx] <= max_distance and min_dist_idx not in used_curr:
            assignments[min_dist_idx] = i
            used_prev.add(i)
            used_curr.add(min_dist_idx)
    
    # Second pass: assign remaining players to new IDs
    next_id = max(assignments) + 1 if assignments else 0
    for i in range(len(assignments)):
        if assignments[i] == -1:
            assignments[i] = next_id
            next_id += 1
    
    return assignments


def process_frame(frame_number, prev_centers=None):
    """Process a single frame for pose estimation and triangulation"""
    # ... (Keep existing code for reading frames and setting up MediaPipe)
    file1 = f"HPUP_033_1_1_1_L_CAM-05_{frame_number:07d}.jpeg"
    file2 = f"HPUP_033_1_1_1_L_CAM-02_{frame_number:07d}.jpeg"
    
    # Read frames
    frame1 = cv2.imread(os.path.join(c233_path, file1))
    frame2 = cv2.imread(os.path.join(c235_path, file2))
    
    if frame1 is None or frame2 is None:
        print(f"Frame {frame_number}: Files not found")
        return None, None
        
    print(f"Processing frame {frame_number}")
    
    # Make copies for visualization
    vis_frame1 = frame1.copy()
    vis_frame2 = frame2.copy()
    
    # Process with MediaPipe
    try:
        results_pose1 = pose.process(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
        results_pose2 = pose.process(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))

        # Check for multiple poses
        poses1 = []
        if results_pose1.pose_landmarks:
            poses1 = [results_pose1.pose_landmarks]

        poses2 = []
        if results_pose2.pose_landmarks:
            poses2 = [results_pose2.pose_landmarks]

        # Filter out None poses
        poses1 = [p for p in poses1 if p is not None]
        poses2 = [p for p in poses2 if p is not None]


        print(f"Frame {frame_number}: Found {len(poses1)} poses in Camera 1, {len(poses2)} poses in Camera 2")

        # Get centers for player matching
        curr_centers1 = [get_pose_center(pose, frame1.shape) for pose in poses1]
        curr_centers2 = [get_pose_center(pose, frame2.shape) for pose in poses2]

        # Match players between frames and cameras
        player_assignments = match_players(prev_centers, curr_centers1, curr_centers2)

        # Draw landmarks on visualization frames
        for pose_landmarks in poses1:
            for i, landmark in enumerate(pose_landmarks.landmark):     # Access landmark data
                x = landmark.x * frame1.shape[1]
                y = landmark.y * frame1.shape[0]
                mp_drawing.draw_landmarks(vis_frame1, pose_landmarks, mp_pose.POSE_CONNECTIONS)
        for pose_landmarks in poses2:
             for i, landmark in enumerate(pose_landmarks.landmark):     # Access landmark data
                x = landmark.x * frame2.shape[1]
                y = landmark.y * frame2.shape[0]
                mp_drawing.draw_landmarks(vis_frame2, pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Save visualization frames
        cv2.imwrite(os.path.join(output_dir, f"cam1_frame{frame_number}.jpg"), vis_frame1)
        cv2.imwrite(os.path.join(output_dir, f"cam2_frame{frame_number}.jpg"), vis_frame2)

        # Process each detected pose
        frame_data = []
        for player_idx, (pose1, pose2) in enumerate(zip(poses1, poses2)):
            points_3d = []
            for i in range(33):  # MediaPipe detects 33 landmarks
                try:
                    point1 = np.array([
                        pose1.landmark[i].x * frame1.shape[1],
                        pose1.landmark[i].y * frame1.shape[0]
                    ])
                    point2 = np.array([
                        pose2.landmark[i].x * frame2.shape[1],
                        pose2.landmark[i].y * frame2.shape[0]
                    ])
                    point_3d = triangulate_point(P1, P2, point1, point2)
                    points_3d.append(point_3d)
                except Exception as e:
                    print(f"Frame {frame_number}: Error triangulating point {i} for player {player_idx}")
                    points_3d.append(np.zeros(3))

            points_3d = np.array(points_3d)
            frame_data.append({
                'frame': frame_number,
                'player_id': player_assignments[player_idx],
                'points': points_3d
            })

        return frame_data, curr_centers1 + curr_centers2

    except Exception as e:
        print(f"Frame {frame_number}: Error in processing: {e}")
        return None, None

def match_players(prev_centers, curr_centers1, curr_centers2, max_distance=100):
    """Match players between frames and cameras based on their center positions"""
    curr_centers = curr_centers1 + curr_centers2
    if not prev_centers or not curr_centers:
        return list(range(len(curr_centers)))

    distances = cdist(prev_centers, curr_centers)
    assignments = [-1] * len(curr_centers)
    used_prev = set()
    used_curr = set()

    for i in range(len(prev_centers)):
        if i in used_prev:
            continue
        min_dist_idx = np.argmin(distances[i])
        if distances[i][min_dist_idx] <= max_distance and min_dist_idx not in used_curr:
            assignments[min_dist_idx] = i
            used_prev.add(i)
            used_curr.add(min_dist_idx)

    next_id = max(assignments) + 1 if assignments else 0
    for i in range(len(assignments)):
        if assignments[i] == -1:
            assignments[i] = next_id
            next_id += 1

    return assignments




def draw_skeleton_3d(ax, points, color='blue', alpha=0.7, linewidth=2):
    """Draw a 3D skeleton connecting landmarks according to SKELETON_CONNECTIONS"""
    if points.ndim != 2 or points.shape[1] != 3:
        print(f"Invalid points array shape: {points.shape}")
        return
    
    # Draw points with proper color handling
    if isinstance(color, (list, tuple, np.ndarray)) and len(color) in (3, 4):
        color = np.array(color).reshape(1, -1)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=color, s=20, alpha=alpha)
    
    # Draw connections
    for connection in SKELETON_CONNECTIONS:
        idx1, idx2 = connection
        if idx1 < len(points) and idx2 < len(points):
            # Get points coordinates
            x1, y1, z1 = points[idx1]
            x2, y2, z2 = points[idx2]
            
            # Skip if either point is zero (missing landmark)
            if np.all(np.isclose([x1, y1, z1], 0)) or np.all(np.isclose([x2, y2, z2], 0)):
                continue
            
            # Draw line
            ax.plot([x1, x2], [y1, y2], [z1, z2], color=color, linewidth=linewidth, alpha=alpha)

def create_animation(all_3d_points):
    """Create a 3D animation from the collected 3D points for each player"""
    if not all_3d_points:
        print("No 3D points to animate")
        return

    print(f"Creating animation with {len(all_3d_points)} frames...")

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Find the bounds of all points to set consistent axis limits
    all_points = []
    for frame_data in all_3d_points:
        for player_data in frame_data:
            valid_points = player_data['points'][~np.all(player_data['points'] == 0, axis=1)]
            if len(valid_points) > 0:
                all_points.append(valid_points)
    
    if not all_points:
        print("No valid points found for animation")
        return
    
    all_points = np.vstack(all_points)
    x_min, y_min, z_min = np.min(all_points, axis=0)
    x_max, y_max, z_max = np.max(all_points, axis=0)
    padding = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    def update(frame):
        ax.clear()
        for player_data in all_3d_points[frame]:
            color = plt.cm.Set1(player_data['player_id'] % 10)  # Use modulo to cycle through colors
            draw_skeleton_3d(ax, player_data['points'], color=color, alpha=0.8, linewidth=2)

        ax.set_title(f'3D Cricket Players Pose - Frame {all_3d_points[frame][0]["frame"]}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([x_min - padding * x_range, x_max + padding * x_range])
        ax.set_ylim([y_min - padding * y_range, y_max + padding * y_range])
        ax.set_zlim([z_min - padding * z_range, z_max + padding * z_range])
        ax.view_init(elev=20, azim=45)
        return ax

    try:
        anim = FuncAnimation(fig, update, frames=len(all_3d_points), interval=200, blit=False)
        output_path = os.path.join(output_dir, '3d_cricket_skeleton.gif')
        anim.save(output_path, writer='pillow', fps=10, dpi=80)
        print(f"Animation saved as '{output_path}'")
    except Exception as e:
        print(f"Error creating animation: {e}")
    finally:
        plt.close(fig)

def create_unity_json(all_3d_points):
    """Create Unity-compatible JSON from 3D points data for each player"""
    if not all_3d_points:
        print("No 3D points to save for Unity")
        return False
    
    print(f"Creating Unity JSON data for {len(all_3d_points)} frames...")
    
    try:
        # Get MediaPipe landmark names
        landmark_names = [name for name in dir(mp_pose.PoseLandmark) if not name.startswith('_')]
        
        # Convert to Unity format
        unity_data = []
        
        for frame_data in all_3d_points:
            frame_number = frame_data[0]['frame']
            frame_unity_data = {
                "frameNumber": frame_number,
                "players": []
            }
            
            for player_data in frame_data:
                player_id = player_data['player_id']
                points = player_data['points']
                
                player_landmarks = {}
                for i, point in enumerate(points):
                    if i < len(landmark_names):
                        # Convert to Unity coordinate system
                        unity_x = float(point[0])
                        unity_y = float(point[2])  # In Unity, Y is up (MediaPipe's Z)
                        unity_z = float(point[1])  # Swap Y and Z for Unity
                        
                        # Only include non-zero points
                        if not (unity_x == 0 and unity_y == 0 and unity_z == 0):
                            player_landmarks[landmark_names[i]] = {
                                "position": [unity_x, unity_y, unity_z],
                                "confidence": 1.0
                            }
                
                frame_unity_data["players"].append({
                    "playerId": player_id,
                    "landmarks": player_landmarks
                })
            
            unity_data.append(frame_unity_data)
        
        # Save full Unity data
        output_file = os.path.join(unity_dir, 'pose_data_for_unity.json')
        with open(output_file, 'w') as f:
            json.dump(unity_data, f, indent=2)
        
        print(f"Saved Unity data to '{output_file}'")
        print(f"File size: {os.path.getsize(output_file)} bytes")
        
        # Save single frame test file
        if unity_data:
            test_file = os.path.join(unity_dir, 'pose_data_test_frame.json')
            with open(test_file, 'w') as f:
                json.dump([unity_data[0]], f, indent=2)
            
            print(f"Saved test frame to '{test_file}'")
        
        return True
    except Exception as e:
        print(f"Error creating Unity JSON: {e}")
        return False

def save_3d_points(all_3d_points):
    """Save 3D points data to a JSON file for each player"""
    if not all_3d_points:
        print("No 3D points to save")
        return
    
    try:
        # Convert numpy arrays to lists for JSON serialization
        json_data = []
        for frame_data in all_3d_points:
            frame_json = {
                'frame': int(frame_data[0]['frame']),
                'players': []
            }
            
            for player_data in frame_data:
                frame_json['players'].append({
                    'player_id': int(player_data['player_id']),
                    'points': player_data['points'].tolist()
                })
            
            json_data.append(frame_json)
        
        # Save to file
        output_file = os.path.join(output_dir, '3d_points.json')
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"3D points saved to '{output_file}'")
    except Exception as e:
        print(f"Error saving 3D points: {e}")

def main():
    start_time = time.time()
    
    # Process frames that are known to work well for reliable output
    frame_range = list(range(250,507,5))  # Process every 5th frame for better tracking
    print(f"Processing frames {frame_range[0]} to {frame_range[-1]}...")
    
    # Collect 3D points from frames
    all_3d_points = []
    successful_frames = 0
    prev_centers = None
    
    for frame_number in frame_range:
        try:
            result, curr_centers = process_frame(frame_number, prev_centers)
            if result is not None:
                all_3d_points.append(result)
                successful_frames += 1
                prev_centers = curr_centers
        except Exception as e:
            print(f"Error processing frame {frame_number}: {e}")
    
    print(f"Successfully processed {successful_frames} frames with valid 3D points")
    
    if all_3d_points:
        # Save 3D points data to JSON
        save_3d_points(all_3d_points)
        
        # Create a static visualization for verification
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Use different colors for different players
        max_players = max(len(frame_data) for frame_data in all_3d_points)
        colors = plt.cm.Set1(np.linspace(0, 1, max_players))
        
        for frame_data in all_3d_points:
            for player_data in frame_data:
                player_id = player_data['player_id']
                if player_id < len(colors):  # Ensure we have a color for this player
                    draw_skeleton_3d(ax, player_data['points'], color=colors[player_id], alpha=0.5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Cricket Players Pose - All Frames')
        ax.view_init(elev=20, azim=45)
        
        static_file = os.path.join(output_dir, '3d_cricket_static.png')
        plt.savefig(static_file)
        print(f"Static visualization saved as '{static_file}'")
        plt.close(fig)
        
        # Create animation
        create_animation(all_3d_points)
        
        # Create Unity JSON data
        create_unity_json(all_3d_points)
        
        # Print summary
        elapsed_time = time.time() - start_time
        print(f"\nProcessing complete:")
        print(f"- Processed {len(frame_range)} frames")
        print(f"- Successful frames: {successful_frames}")
        print(f"- Created 3D skeleton visualization")
        print(f"- Generated Unity-compatible JSON data")
        print(f"- Total processing time: {elapsed_time:.2f} seconds")
    else:
        print("No frames with valid 3D points were found")
    
    # Clean up
    pose.close()

if __name__ == "__main__":
    print("3D Cricket Pose Processing Script")
    print("=================================")
    main()