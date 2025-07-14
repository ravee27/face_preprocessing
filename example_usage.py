#!/usr/bin/env python3
"""
Example usage of the FacePreprocessor class for both regular face images and document images.
"""

import os
import sys
from main import FacePreprocessor


def process_image(image_path: str, output_dir: str = "aligned_faces"):
    """
    Process a single image (face or document) and extract the aligned face.
    
    Args:
        image_path (str): Path to the input image
        output_dir (str): Directory to save the output
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return False
    
    # Initialize preprocessor
    preprocessor = FacePreprocessor(device='cpu')
    
    print(f"\n{'='*60}")
    print(f"Processing: {image_path}")
    print(f"{'='*60}")
    
    try:
        # Process the image
        aligned_face = preprocessor.process_face(image_path)
        
        if aligned_face is not None:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"aligned_{base_name}.jpg")
            
            # Save the aligned face
            if preprocessor.save_image(aligned_face, output_path):
                print(f"✅ Successfully processed and saved aligned face to: {output_path}")
                return True
            else:
                print("❌ Failed to save aligned face")
                return False
        else:
            print("❌ Failed to process face")
            return False
            
    except ValueError as e:
        print(f"❌ Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def main():
    """Main function to demonstrate usage."""
    print("Face Preprocessor - Document and Face Image Processing")
    print("=" * 60)
    
    # Example usage with different types of images
    test_images = [
        # Regular face images
        "/path/to/regular_face_image.jpg",
        
        # Document images (passports, driving licenses, etc.)
        "/path/to/passport_image.jpg",
        "/path/to/driving_license_image.jpg",
        "/path/to/id_card_image.jpg",
        
        # Add your actual image paths here
    ]
    
    # Filter out non-existent paths for demonstration
    existing_images = [img for img in test_images if os.path.exists(img)]
    
    if not existing_images:
        print("No existing test images found.")
        print("Please update the test_images list with actual image paths.")
        print("\nExample usage:")
        print("python example_usage.py /path/to/your/image.jpg")
        return
    
    print(f"Found {len(existing_images)} test image(s)")
    
    # Process each image
    success_count = 0
    for image_path in existing_images:
        if process_image(image_path):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Processing complete: {success_count}/{len(existing_images)} images processed successfully")
    print(f"{'='*60}")


if __name__ == "__main__":
    # If command line argument provided, process that specific image
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if process_image(image_path):
            print("✅ Image processed successfully!")
        else:
            print("❌ Failed to process image")
            sys.exit(1)
    else:
        # Run with example images
        main() 