Overview

This script processes cricket player video frames to:

Extract pose landmarks using MediaPipe

Triangulate 3D points from two camera views

Generate 3D skeleton visualizations (static and animated)

Create Unity-compatible JSON files with the 3D pose data

Convert JSON output into a 3D render for Unreal Engine, Unity, or Blender

Requirements

Ensure you have the following dependencies installed before running the script:

pip install opencv-python mediapipe numpy matplotlib
----------------------------------------------------------------------------
File Structure

.
├── 233_im/                # Camera 1 image frames
├── 235_im/                # Camera 2 image frames
├── intrinsic.json         # Camera intrinsic parameters
├── extrinsic.json         # Camera extrinsic parameters
├── output/                # Processed outputs (images, data)
├── unity_data/            # Unity-compatible JSON files
└── cricket_3d_pose.py     # Main script
-----------------------------------------------------------------------------
How It Works

1. Camera Calibration Loading

The script reads camera intrinsic (focal length, optical center) and extrinsic (rotation, translation) parameters from intrinsic.json and extrinsic.json.

It computes the projection matrices for both cameras.

2. Pose Detection

Uses MediaPipe to detect 2D landmarks from images captured from both camera views.

Saves detected landmarks on images for visualization.

3. 3D Point Triangulation

If landmarks are detected in both views, the script triangulates corresponding 3D points.

The results are stored in a structured format.

4. Data Storage

Triangulated 3D points are saved in JSON format for easy integration with Unity, Unreal Engine, and Blender.

Output images with pose annotations are stored in the output/ folder.

Running the Script

To process a specific frame, use:

python cricket_3d_pose.py 

To process multiple frames, modify the script to iterate over a range.

Output Format

Annotated Images: Stored in output/, showing detected landmarks.

3D Pose JSON: Stored in unity_data/, formatted as:

{
  "frame": 123,
  "points": [[x1, y1, z1], [x2, y2, z2], ..., [x33, y33, z33]]
}
---------------------------------------------------------
Core Logic

1)The system operates by processing two separate camera inputs, each with its own rotation vectors and camera matrices. The process flows as follows:

2)Camera Input: We start with two different camera feeds, each with rotation vectors and camera matrices that describe their orientation and position in 3D space.

3)Segmentation and Joint Mapping: The camera inputs are fed into a MediaPipe model, which segments individuals and maps out their joints (such as elbows, knees, and shoulders).

4)3D Conversion: The joint data from MediaPipe is sent to another model based on AXIS 3D, where it is converted into 3D output data.
-------------------------------------------------------
Rendering: The data from the first few frames (typically frames 1 through 5 or 6) is rendered at once, creating a dynamic 3D visualization of the joint movements. This visualization is saved as a GIF using Matplotlib, capturing the motion across the frames.

JSON Output: Alongside the 3D render, a JSON file is generated that contains the 3D coordinates of all the joints across all frames. This JSON file can be imported into 3D software like Unity, Blender, or Unreal Engine, making the output highly compatible with other models and useful for feature scaling.

--------------------------------------------------------
