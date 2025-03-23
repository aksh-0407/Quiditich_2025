# Cricket 3D Pose Estimation

This project processes cricket player video frames to extract pose landmarks and generate 3D skeleton visualizations. It leverages MediaPipe for pose detection, triangulates 3D points using two camera views, and produces Unity-compatible JSON files for use in 3D software such as Unity, Unreal Engine, and Blender.

## Overview

This script provides a comprehensive solution for:

- **Pose Landmark Extraction**: Uses MediaPipe to extract 2D landmarks from video frames.
- **3D Pose Generation**: Triangulates 3D points from two camera views and generates 3D skeleton visualizations.
- **JSON Export**: Saves 3D pose data in JSON format, compatible with Unity, Unreal Engine, and Blender.
- **3D Rendering**: Renders dynamic 3D visualizations of joint movements and saves them as GIFs for easy inspection.

## Requirements

Make sure the following dependencies are installed before running the script:

```bash
pip install opencv-python mediapipe numpy matplotlib
```

### File Structure

```
.
├── 233_im/                # Camera 1 image frames
├── 235_im/                # Camera 2 image frames
├── intrinsic.json         # Camera intrinsic parameters
├── extrinsic.json         # Camera extrinsic parameters
├── output/                # Processed outputs (images, data)
├── unity_data/            # Unity-compatible JSON files
└── cricket_3d_pose.py     # Main script
```

### Input Files

1. **233_im/**: Contains the image frames from Camera 1.
2. **235_im/**: Contains the image frames from Camera 2.
3. **intrinsic.json**: Contains the camera intrinsic parameters (focal length, optical center).
4. **extrinsic.json**: Contains the camera extrinsic parameters (rotation, translation).

### Output Files

1. **Output Images**: Annotated images with detected landmarks will be stored in the `output/` folder.
2. **3D Pose JSON**: JSON files containing the 3D coordinates of joints for each frame will be saved in the `unity_data/` folder. The format is as follows:

```json
{
  "frame": 123,
  "points": [[x1, y1, z1], [x2, y2, z2], ..., [x33, y33, z33]]
}
```

## How It Works

### 1. Camera Calibration Loading

The script begins by loading the camera intrinsic and extrinsic parameters from the `intrinsic.json` and `extrinsic.json` files. It then calculates the projection matrices for both cameras, essential for triangulating 3D points from the 2D landmarks.

### 2. Pose Detection

The script utilizes MediaPipe to detect 2D landmarks from the frames captured by both cameras. These 2D coordinates are saved in the output images for visualization and further analysis.

### 3. 3D Point Triangulation

When landmarks are detected in both camera views, the script triangulates the corresponding 3D points. This step is crucial for converting the 2D pose data into a 3D representation. The results are saved in a structured format for easy use in 3D applications.

### 4. Data Storage

The triangulated 3D points are saved in JSON files within the `unity_data/` folder. These JSON files can be imported into 3D software such as Unity, Unreal Engine, or Blender. Output images with annotated landmarks are stored in the `output/` folder.

## Running the Script

To process a specific frame, execute the script as follows:

```bash
python cricket_3d_pose.py
```

To process multiple frames, modify the script to iterate over a range of frames.

## Rendering 3D Pose

After processing the frames, the script generates a dynamic 3D visualization of joint movements. This visualization is rendered over several frames (typically frames 1–5 or 6) and saved as a GIF using Matplotlib. This helps visualize the player's motion over time.

### Example Command for Rendering

```bash
python cricket_3d_pose.py --render
```

This will output a GIF animation of the 3D pose and save it to the `output/` folder.

## JSON Output

The script generates a JSON file for each frame containing the 3D coordinates of all the detected joints across the frames. This output is compatible with 3D engines like Unity, Unreal Engine, and Blender, making it easy to import into these tools for further development.

Example JSON Output:

```json
{
  "frame": 123,
  "points": [
    [x1, y1, z1], 
    [x2, y2, z2], 
    ...
    [x33, y33, z33]
  ]
}
```

### Example Directory Structure After Running the Script

```
.
├── output/
│   ├── frame_001.png        # Image with pose annotations for frame 1
│   ├── frame_002.png        # Image with pose annotations for frame 2
│   └── animation.gif        # 3D Pose animation GIF
├── unity_data/
│   ├── frame_001.json       # 3D pose data for frame 1
│   ├── frame_002.json       # 3D pose data for frame 2
│   └── frame_003.json       # 3D pose data for frame 3
└── cricket_3d_pose.py       # Main script
```

## Compatibility

The generated JSON files can be easily imported into 3D software like:

- **Unity**
- **Unreal Engine**
- **Blender**

These files contain the 3D coordinates of all detected joints across the frames, making it easy to visualize and animate the player's movements.


---

Enjoy using this script to analyze cricket player poses in 3D!
