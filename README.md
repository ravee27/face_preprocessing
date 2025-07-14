# Face Preprocessor

A Python library for preprocessing face images, including rotation, alignment, and face detection. The library can handle both regular face images and document images (passports, driving licenses, ID cards, etc.).

## Features

- **Face Detection**: Detects human faces in images using advanced face alignment techniques
- **Document Processing**: Automatically detects and extracts faces from document images
- **Face Alignment**: Aligns faces based on eye positions for consistent processing
- **Auto-rotation**: Automatically rotates images to make faces upright
- **Multiple Detection Methods**: Uses both face_alignment and OpenCV for robust face detection
- **Error Handling**: Comprehensive error handling for various scenarios

## Installation

### Prerequisites

- Python 3.7+
- OpenCV
- NumPy
- face_alignment
- matplotlib (for visualization)

### Install Dependencies

```bash
pip install opencv-python numpy face-alignment matplotlib
```

## Usage

### Basic Usage

```python
from main import FacePreprocessor

# Initialize the preprocessor
preprocessor = FacePreprocessor(device='cpu')  # or 'cuda' for GPU

# Process a regular face image
aligned_face = preprocessor.process_face("path/to/face_image.jpg")

# Process a document image (passport, driving license, etc.)
aligned_face = preprocessor.process_face("path/to/passport_image.jpg")

# Save the processed face
if aligned_face is not None:
    preprocessor.save_image(aligned_face, "output/aligned_face.jpg")
```

### Command Line Usage

```bash
# Process a single image
python example_usage.py /path/to/your/image.jpg

# Run with example images (update paths in the script first)
python example_usage.py
```

## Document Image Processing

The library automatically detects if an image is a document (passport, driving license, ID card) and extracts the face region before processing. Document detection is based on:

- **Aspect Ratio**: Documents are typically rectangular (aspect ratio 1.2-2.5)
- **Edge Density**: Documents have high edge density due to text and borders
- **Straight Lines**: Documents typically have straight borders

### Document Processing Flow

1. **Document Detection**: Automatically detects if the image is a document
2. **Face Extraction**: Extracts face regions from the document using multiple detection methods
3. **Face Processing**: Processes the extracted face using the standard pipeline
4. **Error Handling**: Throws clear error messages if no face is detected in documents

### Error Handling

The library provides specific error handling for document images:

```python
try:
    aligned_face = preprocessor.process_face("document_image.jpg")
except ValueError as e:
    print(f"Document processing error: {e}")
    # Error: "No human face detected in the document image. Please provide an image with a clear human face."
```

## API Reference

### FacePreprocessor Class

#### `__init__(device='cpu')`
Initialize the face preprocessor.

**Parameters:**
- `device` (str): Device to run face alignment on ('cpu' or 'cuda')

#### `process_face(image_path, output_size=(256, 256))`
Complete face processing pipeline for both regular and document images.

**Parameters:**
- `image_path` (str): Path to input image
- `output_size` (tuple): Desired output size (width, height)

**Returns:**
- `np.ndarray` or `None`: Processed face image or None if processing fails

**Raises:**
- `ValueError`: If no human face is detected in document images

#### `is_document_image(image)`
Detect if the image is likely a document.

**Parameters:**
- `image` (np.ndarray): Input image

**Returns:**
- `bool`: True if image appears to be a document

#### `extract_face_from_document(image)`
Extract face region from a document image.

**Parameters:**
- `image` (np.ndarray): Document image

**Returns:**
- `np.ndarray` or `None`: Extracted face region or None if no face found

## Examples

### Processing Regular Face Images

```python
from main import FacePreprocessor

preprocessor = FacePreprocessor()

# Process a regular face image
aligned_face = preprocessor.process_face("face.jpg")
if aligned_face is not None:
    preprocessor.save_image(aligned_face, "aligned_face.jpg")
```

### Processing Document Images

```python
from main import FacePreprocessor

preprocessor = FacePreprocessor()

# Process a passport image
try:
    aligned_face = preprocessor.process_face("passport.jpg")
    if aligned_face is not None:
        preprocessor.save_image(aligned_face, "passport_face.jpg")
except ValueError as e:
    print(f"No face found in document: {e}")
```

### Batch Processing

```python
from main import FacePreprocessor
import os

preprocessor = FacePreprocessor()

image_paths = [
    "face1.jpg",
    "passport1.jpg", 
    "driving_license.jpg"
]

for image_path in image_paths:
    try:
        aligned_face = preprocessor.process_face(image_path)
        if aligned_face is not None:
            output_path = f"aligned_{os.path.basename(image_path)}"
            preprocessor.save_image(aligned_face, output_path)
            print(f"Processed: {image_path}")
    except ValueError as e:
        print(f"Error processing {image_path}: {e}")
```

## Supported Document Types

- **Passports**: National and international passports
- **Driving Licenses**: Various driving license formats
- **ID Cards**: National identity cards
- **Other Documents**: Any document with a clear human face

## Error Messages

- `"No human face detected in the document image. Please provide an image with a clear human face."` - When no face is found in a document
- `"Image file not found: {path}"` - When the image file doesn't exist
- `"Could not load image from: {path}"` - When the image file is corrupted or unsupported

## Tips for Best Results

1. **Image Quality**: Use clear, well-lit images
2. **Face Visibility**: Ensure the face is clearly visible and not obscured
3. **Document Orientation**: Documents should be properly oriented
4. **Face Size**: The face should be reasonably sized in the image
5. **Multiple Faces**: The library processes the largest detected face

## Troubleshooting

### Common Issues

1. **No faces detected in document**
   - Ensure the document has a clear, unobstructed face
   - Check image quality and lighting
   - Try different document orientations

2. **OpenCV cascade classifier not found**
   - The library will fall back to face_alignment only
   - Install OpenCV with cascade files: `pip install opencv-contrib-python`

3. **Memory issues with large images**
   - Resize large images before processing
   - Use smaller output sizes

### Performance Optimization

- Use GPU (`device='cuda'`) for faster processing
- Process images in batches for efficiency
- Consider image resizing for very large images 