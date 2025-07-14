#!/usr/bin/env python3
"""
Example usage of the FacePreprocessor class.
This script demonstrates various ways to use the face preprocessing functionality.
"""

from main import FacePreprocessor
import matplotlib.pyplot as plt
import os


def example_basic_usage(image_path):
    """Basic usage example."""
    print("=== Basic Usage Example ===")
    
    # Initialize preprocessor
    preprocessor = FacePreprocessor(device='cpu')
    
    # Example image path (update this to your image path)
    image_path = image_path
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        print("Please update the image_path variable with a valid image path.")
        return
    
    # Process face with default settings
    aligned_face = preprocessor.process_face(image_path)
    
    if aligned_face is not None:
        # Save the processed face
        output_path = "aligned_faces/processed_face.jpg"
        preprocessor.save_image(aligned_face, output_path)
        print(f"Face processed and saved to: {output_path}")
    else:
        print("Failed to process face")


def example_custom_settings(image_path):
    """Example with custom settings."""
    print("\n=== Custom Settings Example ===")
    
    preprocessor = FacePreprocessor(device='cpu')
    
    # Example image path
    image_path = image_path
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    try:
        # Load image
        image = preprocessor.load_image(image_path)
        print(f"Loaded image shape: {image.shape}")
        
        # Detect faces
        landmarks_list = preprocessor.detect_faces(image)
        
        if landmarks_list:
            # Process each detected face
            for i, landmarks in enumerate(landmarks_list):
                print(f"\nProcessing face {i+1}:")
                
                # Auto-upright the image
                upright_image, upright_landmarks = preprocessor.auto_upright_image(image)
                
                # Align with custom size
                aligned_face, _ = preprocessor.align_face(
                    upright_image, 
                    upright_landmarks, 
                    output_size=(512, 512)  # Custom size
                )
                
                # Save with custom name
                output_path = f"aligned_faces/face_{i+1}_512x512.jpg"
                preprocessor.save_image(aligned_face, output_path)
                print(f"Face {i+1} saved to: {output_path}")
        else:
            print("No faces detected in the image")
            
    except Exception as e:
        print(f"Error: {e}")


def example_rotation_only(image_path):
    """Example showing just rotation functionality."""
    print("\n=== Rotation Example ===")
    
    preprocessor = FacePreprocessor()
    
    image_path = image_path
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    try:
        # Load image
        image = preprocessor.load_image(image_path)
        
        # Rotate by 45 degrees
        rotated_image, _ = preprocessor.rotate_image(image, 45)
        
        # Save rotated image
        output_path = "aligned_faces/rotated_45_degrees.jpg"
        preprocessor.save_image(rotated_image, output_path)
        print(f"Rotated image saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")


def example_batch_processing():
    """Example of batch processing multiple images."""
    print("\n=== Batch Processing Example ===")
    
    preprocessor = FacePreprocessor()
    
    # List of image paths (update these with your actual image paths)
    image_paths = [
        "path/to/image1.jpg",
        "path/to/image2.jpg",
        "path/to/image3.jpg"
    ]
    
    processed_count = 0
    
    for i, image_path in enumerate(image_paths):
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        
        print(f"\nProcessing image {i+1}/{len(image_paths)}: {image_path}")
        
        try:
            aligned_face = preprocessor.process_face(image_path)
            
            if aligned_face is not None:
                output_path = f"aligned_faces/batch_face_{i+1}.jpg"
                preprocessor.save_image(aligned_face, output_path)
                processed_count += 1
                print(f"Successfully processed: {output_path}")
            else:
                print(f"Failed to process: {image_path}")
                
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    print(f"\nBatch processing complete. Successfully processed {processed_count}/{len(image_paths)} images.")


def example_with_visualization(image_path):
    """Example with matplotlib visualization."""
    print("\n=== Visualization Example ===")
    
    preprocessor = FacePreprocessor()
    
    image_path = image_path
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    try:
        # Load original image
        original = preprocessor.load_image(image_path)
        
        # Process face
        aligned_face = preprocessor.process_face(image_path)
        
        if aligned_face is not None:
            # Create visualization
            plt.figure(figsize=(15, 5))
            
            # Original image
            plt.subplot(1, 3, 1)
            plt.title("Original Image")
            plt.imshow(original)
            plt.axis("off")
            
            # Aligned face
            plt.subplot(1, 3, 2)
            plt.title("Aligned Face")
            plt.imshow(aligned_face)
            plt.axis("off")
            
            # Save processed face
            output_path = "aligned_faces/visualized_face.jpg"
            preprocessor.save_image(aligned_face, output_path)
            
            plt.subplot(1, 3, 3)
            plt.title("Saved Image Path")
            plt.text(0.1, 0.5, f"Saved to:\n{output_path}", 
                    transform=plt.gca().transAxes, fontsize=12)
            plt.axis("off")
            
            plt.tight_layout()
            plt.savefig("aligned_faces/comparison.png", dpi=150, bbox_inches='tight')
            plt.show()
            
            print(f"Visualization saved to: aligned_faces/comparison.png")
        else:
            print("Failed to process face for visualization")
            
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Run all examples."""
    print("FacePreprocessor Examples")
    print("=" * 50)
    
    # Create output directory
    os.makedirs("aligned_faces", exist_ok=True)
    
    # Run examples (comment out the ones you don't want to run)
    example_basic_usage()
    example_custom_settings()
    example_rotation_only()
    example_batch_processing()
    example_with_visualization()
    
    print("\nAll examples completed!")


if __name__ == "__main__":
    main() 