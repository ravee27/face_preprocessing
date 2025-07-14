# Face Preprocessing Library

A comprehensive Python library for face image preprocessing, including rotation, alignment, and face detection using facial landmarks.

## Features

- **Face Detection**: Detect faces in images using facial landmarks
- **Auto-rotation**: Automatically rotate images to make faces upright
- **Face Alignment**: Align faces based on eye positions
- **Image Cropping**: Crop and resize faces to specified dimensions
- **Batch Processing**: Process multiple images efficiently
- **Visualization**: Display and save comparison images

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. For GPU support (optional), install PyTorch with CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### Basic Usage

```python
from main import FacePreprocessor

# Initialize the preprocessor
preprocessor = FacePreprocessor(device='cpu')  # or 'cuda' for GPU

# Process a single image
image_path = "path/to/your/image.jpg"
aligned_face = preprocessor.process_face(image_path)

if aligned_face is not None:
    # Save the processed face
    preprocessor.save_image(aligned_face, "aligned_faces/processed_face.jpg")
    print("Face processed successfully!")
else:
    print("Failed to process face")
```

### Advanced Usage

```python
from main import FacePreprocessor
import matplotlib.pyplot as plt

preprocessor = FacePreprocessor()

# Load image
image = preprocessor.load_image("path/to/image.jpg")

# Detect faces
landmarks_list = preprocessor.detect_faces(image)

if landmarks_list:
    for i, landmarks in enumerate(landmarks_list):
        # Auto-upright the image
        upright_image, upright_landmarks = preprocessor.auto_upright_image(image)
        
        # Align with custom size
        aligned_face, _ = preprocessor.align_face(
            upright_image, 
            upright_landmarks, 
            output_size=(512, 512)
        )
        
        # Save with custom name
        preprocessor.save_image(aligned_face, f"aligned_faces/face_{i+1}.jpg")
```

## Class Methods

### FacePreprocessor

The main class for face preprocessing operations.

#### `__init__(device='cpu')`
Initialize the preprocessor with face alignment model.

**Parameters:**
- `device` (str): Device to run face alignment on ('cpu' or 'cuda')

#### `load_image(image_path)`
Load and preprocess an image from file path.

**Parameters:**
- `image_path` (str): Path to the image file

**Returns:**
- `np.ndarray`: Loaded image in RGB format

#### `rotate_image(image, angle)`
Rotate the image around its center by the given angle.

**Parameters:**
- `image` (np.ndarray): Input image
- `angle` (float): Rotation angle in degrees

**Returns:**
- `Tuple[np.ndarray, np.ndarray]`: Rotated image and rotation matrix

#### `detect_faces(image)`
Detect faces in the image and return landmarks.

**Parameters:**
- `image` (np.ndarray): Input image

**Returns:**
- `Optional[List[np.ndarray]]`: List of face landmarks or None if no faces detected

#### `auto_upright_image(image, max_attempts=10)`
Automatically rotate image to make face upright.

**Parameters:**
- `image` (np.ndarray): Input image
- `max_attempts` (int): Maximum number of rotation attempts

**Returns:**
- `Tuple[np.ndarray, np.ndarray]`: Upright image and detected landmarks

#### `align_face(image, landmarks, output_size=(256, 256))`
Align face based on eye positions and crop to specified size.

**Parameters:**
- `image` (np.ndarray): Input image
- `landmarks` (np.ndarray): Face landmarks
- `output_size` (Tuple[int, int]): Desired output size (width, height)

**Returns:**
- `Tuple[np.ndarray, np.ndarray]`: Aligned face and rotated image

#### `process_face(image_path, output_size=(256, 256))`
Complete face processing pipeline: load, detect, align, and crop face.

**Parameters:**
- `image_path` (str): Path to input image
- `output_size` (Tuple[int, int]): Desired output size

**Returns:**
- `Optional[np.ndarray]`: Processed face image or None if processing fails

#### `save_image(image, output_path)`
Save image to file.

**Parameters:**
- `image` (np.ndarray): Image to save
- `output_path` (str): Output file path

**Returns:**
- `bool`: True if saved successfully, False otherwise

## Examples

### Example 1: Basic Face Processing

```python
from main import FacePreprocessor

preprocessor = FacePreprocessor()
aligned_face = preprocessor.process_face("input.jpg")
if aligned_face is not None:
    preprocessor.save_image(aligned_face, "output.jpg")
```

### Example 2: Batch Processing

```python
from main import FacePreprocessor
import os

preprocessor = FacePreprocessor()
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]

for i, path in enumerate(image_paths):
    if os.path.exists(path):
        aligned_face = preprocessor.process_face(path)
        if aligned_face is not None:
            preprocessor.save_image(aligned_face, f"aligned_face_{i+1}.jpg")
```

### Example 3: Custom Settings

```python
from main import FacePreprocessor

preprocessor = FacePreprocessor(device='cuda')  # Use GPU

# Load and process with custom size
image = preprocessor.load_image("input.jpg")
landmarks_list = preprocessor.detect_faces(image)

if landmarks_list:
    upright_image, upright_landmarks = preprocessor.auto_upright_image(image)
    aligned_face, _ = preprocessor.align_face(
        upright_image, 
        upright_landmarks, 
        output_size=(512, 512)
    )
    preprocessor.save_image(aligned_face, "custom_size_face.jpg")
```

## Running Examples

Run the example script to see various usage patterns:

```bash
python example_usage.py
```

This will demonstrate:
- Basic usage
- Custom settings
- Rotation only
- Batch processing
- Visualization

## Dependencies

- `opencv-python`: Image processing
- `numpy`: Numerical operations
- `face-alignment`: Face landmark detection
- `matplotlib`: Visualization
- `dlib`: Face detection backend
- `torch`: PyTorch for face alignment
- `torchvision`: PyTorch vision utilities

## Notes

- The library uses the `face-alignment` library for facial landmark detection
- Default output size is 256x256 pixels
- Images are automatically converted to RGB format
- The auto-upright feature rotates images in 35-degree increments
- All processed images are saved in BGR format (OpenCV standard)

## Troubleshooting

1. **No faces detected**: Ensure the image contains clear, visible faces
2. **Memory issues**: Use CPU device for large images or reduce batch size
3. **Installation issues**: Make sure all dependencies are properly installed
4. **CUDA errors**: Install PyTorch with appropriate CUDA version for your GPU

## License

This project is open source and available under the MIT License. 