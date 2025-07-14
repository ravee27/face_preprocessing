import cv2
import numpy as np
import face_alignment
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
import os


class FacePreprocessor:
    """
    A class for preprocessing face images including rotation, alignment, and face detection.
    Can handle both regular face images and document images (passports, driving licenses).
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize the FacePreprocessor with face alignment model.
        
        Args:
            device (str): Device to run face alignment on ('cpu' or 'cuda')
        """
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D, 
            flip_input=False, 
            device=device
        )
        
        # Initialize OpenCV face detector for document processing
        # Try different ways to load the cascade classifier
        cascade_paths = [
            'haarcascade_frontalface_default.xml',
            '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
            '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
        ]
        
        self.face_cascade = None
        for path in cascade_paths:
            try:
                self.face_cascade = cv2.CascadeClassifier(path)
                if not self.face_cascade.empty():
                    break
            except:
                continue
        
        if self.face_cascade is None or self.face_cascade.empty():
            print("Warning: Could not load face cascade classifier. Document face detection may be limited.")
            self.face_cascade = None
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess an image from file path.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            np.ndarray: Loaded image in RGB format
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert RGBA to RGB if necessary (remove alpha channel)
        if image.shape[-1] == 4:
            image = image[:, :, :3]
        
        return image
    
    def is_document_image(self, image: np.ndarray) -> bool:
        """
        Detect if the image is likely a document (passport, driving license, etc.).
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            bool: True if image appears to be a document, False otherwise
        """
        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Check aspect ratio - documents are typically rectangular
        height, width = gray.shape
        aspect_ratio = width / height
        
        # Documents typically have aspect ratios between 1.2 and 2.5
        is_rectangular = 1.2 <= aspect_ratio <= 2.5
        
        # Check for text-like features using edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)
        
        # Documents typically have high edge density due to text and borders
        has_high_edge_density = edge_density > 0.05
        
        # Check for straight lines (document borders)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        has_straight_lines = lines is not None and len(lines) > 10
        
        is_document = is_rectangular and (has_high_edge_density or has_straight_lines)
        
        print(f"Document detection - Aspect ratio: {aspect_ratio:.2f}, "
              f"Edge density: {edge_density:.4f}, "
              f"Is document: {is_document}")
        
        return is_document
    
    def extract_face_from_document(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face region from a document image using multiple detection methods.
        
        Args:
            image (np.ndarray): Document image
            
        Returns:
            Optional[np.ndarray]: Extracted face region or None if no face found
        """
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Method 1: Use OpenCV face cascade (if available)
        faces_cv = []
        if self.face_cascade is not None:
            faces_cv = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
        
        # Method 2: Use face_alignment for more precise detection
        faces_fa = self.fa.get_landmarks(image)
        
        print(f"OpenCV detected {len(faces_cv)} face(s)")
        print(f"Face alignment detected {len(faces_fa) if faces_fa else 0} face(s)")
        
        # If face_alignment found faces, use the largest one
        if faces_fa and len(faces_fa) > 0:
            # Find the face with the largest bounding box
            largest_face = None
            max_area = 0
            
            for landmarks in faces_fa:
                # Calculate bounding box from landmarks
                min_x, min_y = landmarks.min(axis=0)
                max_x, max_y = landmarks.max(axis=0)
                area = (max_x - min_x) * (max_y - min_y)
                
                if area > max_area:
                    max_area = area
                    largest_face = landmarks
            
            if largest_face is not None:
                # Extract face region with padding
                min_x, min_y = largest_face.min(axis=0)
                max_x, max_y = largest_face.max(axis=0)
                
                # Add padding
                padding = 50
                min_x = max(0, int(min_x - padding))
                min_y = max(0, int(min_y - padding))
                max_x = min(image.shape[1], int(max_x + padding))
                max_y = min(image.shape[0], int(max_y + padding))
                
                face_region = image[min_y:max_y, min_x:max_x]
                print(f"Extracted face region: {face_region.shape}")
                return face_region
        
        # Fallback to OpenCV detection if face_alignment failed
        if len(faces_cv) > 0:
            # Use the largest detected face
            largest_face = max(faces_cv, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face
            
            # Add padding
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            
            face_region = image[y:y+h, x:x+w]
            print(f"Extracted face region (OpenCV): {face_region.shape}")
            return face_region
        
        print("No faces detected in document")
        return None
    
    def rotate_image(self, image: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rotate the image around its center by the given angle.
        
        Args:
            image (np.ndarray): Input image
            angle (float): Rotation angle in degrees
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Rotated image and rotation matrix
        """
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rot_matrix, (w, h))
        return rotated, rot_matrix
    
    def rotate_image_from_path(self, image_path: str, angle: float) -> np.ndarray:
        """
        Load and rotate an image from file path by specified angle.
        
        Args:
            image_path (str): Path to the image file
            angle (float): Rotation angle in degrees
            
        Returns:
            np.ndarray: Rotated image
        """
        image = self.load_image(image_path)
        rotated, _ = self.rotate_image(image, angle)
        return rotated
    
    def are_eyes_above_nose(self, landmarks: np.ndarray) -> bool:
        """
        Check if eyes are positioned above the nose in the face landmarks.
        In image coordinates, Y increases downward, so for an upright face:
        - Eyes should be above nose (eye_y < nose_y)
        - Nose should be above mouth (nose_y < mouth_y)
        
        Args:
            landmarks (np.ndarray): Face landmarks array
            
        Returns:
            bool: True if face is upright, False otherwise
        """
        left_eye_y = landmarks[36:42, 1].mean()
        right_eye_y = landmarks[42:48, 1].mean()
        nose_y = landmarks[30, 1]
        mouth_center = landmarks[48:68, 1].mean()
        
        # print(f"Left eye Y: {left_eye_y:.2f}, Right eye Y: {right_eye_y:.2f}, "
        #       f"Nose Y: {nose_y:.2f}, Mouth center: {mouth_center:.2f}")
        
        # For upright face: eyes above nose, nose above mouth
        eyes_above_nose = left_eye_y < nose_y and right_eye_y < nose_y
        nose_above_mouth = nose_y < mouth_center
        
        is_upright = eyes_above_nose and nose_above_mouth
        print(f"Eyes above nose: {eyes_above_nose}, Nose above mouth: {nose_above_mouth}, Upright: {is_upright}")
        
        return is_upright
    
    def auto_upright_image(self, image: np.ndarray, max_attempts: int = 18) -> Tuple[np.ndarray, np.ndarray]:
        """
        Automatically rotate image to make face upright.
        
        Args:
            image (np.ndarray): Input image
            max_attempts (int): Maximum number of rotation attempts (default 18 for 360° coverage)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Upright image and detected landmarks
            
        Raises:
            Exception: If no upright face is detected within rotation attempts
        """
        current_image = image.copy()
        angle = 0
        rotation_step = 20  # Smaller step for more precise detection
        
        for attempt in range(max_attempts):
            preds = self.fa.get_landmarks(current_image)
            print(f"Attempt {attempt + 1}/{max_attempts} at angle: {angle}° - "
                  f"Found {len(preds) if preds else 0} face(s)")
            
            if preds and len(preds) > 0:
                # Check if face is upright
                if self.are_eyes_above_nose(preds[0]):
                    print(f"Face is upright at angle: {angle}°")
                    return current_image, preds[0]
                else:
                    print(f"Face detected but not upright at angle: {angle}°")
            else:
                print(f"No faces detected at angle: {angle}°")
            
            # Rotate image for next attempt
            current_image, _ = self.rotate_image(current_image, rotation_step)
            angle += rotation_step
            
            # Normalize angle to 0-360 range
            angle = angle % 360
        
        raise Exception(f"Failed to detect upright face within {max_attempts} rotation attempts (360° coverage).")
    
    def align_face(self, image: np.ndarray, landmarks: np.ndarray, 
                   output_size: Tuple[int, int] = (256, 256)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align face based on eye positions and crop to specified size.
        
        Args:
            image (np.ndarray): Input image
            landmarks (np.ndarray): Face landmarks
            output_size (Tuple[int, int]): Desired output size (width, height)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Aligned face and rotated image
        """
        # Calculate eye centers
        left_eye_center = landmarks[36:42].mean(axis=0)
        right_eye_center = landmarks[42:48].mean(axis=0)
        
        # Calculate rotation angle to align eyes horizontally
        dx = right_eye_center[0] - left_eye_center[0]
        dy = right_eye_center[1] - left_eye_center[1]
        angle = np.arctan2(dy, dx) * 180 / np.pi
        
        # Calculate eye center for rotation
        eye_center = ((left_eye_center[0] + right_eye_center[0]) / 2,
                      (left_eye_center[1] + right_eye_center[1]) / 2)
        
        print(f"Aligning face with angle: {angle:.2f} degrees")
        
        # Rotate image
        rot_matrix = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
        rotated = cv2.warpAffine(image, rot_matrix, (image.shape[1], image.shape[0]))
        
        # Transform landmarks
        ones = np.ones(shape=(len(landmarks), 1))
        landmarks_ones = np.hstack([landmarks, ones])
        new_landmarks = rot_matrix.dot(landmarks_ones.T).T
        
        # Calculate bounding box
        min_x, min_y = new_landmarks.min(axis=0)
        max_x, max_y = new_landmarks.max(axis=0)
        
        # Add padding and crop
        padding = 50
        min_x = max(0, int(min_x - padding))
        min_y = max(0, int(min_y - padding))
        max_x = min(rotated.shape[1], int(max_x + padding))
        max_y = min(rotated.shape[0], int(max_y + padding))
        
        cropped = rotated[min_y:max_y, min_x:max_x]
        aligned = cv2.resize(cropped, output_size)
        
        return aligned, rotated
    
    def detect_faces(self, image: np.ndarray) -> Optional[List[np.ndarray]]:
        """
        Detect faces in the image and return landmarks.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            Optional[List[np.ndarray]]: List of face landmarks or None if no faces detected
        """
        preds = self.fa.get_landmarks(image)
        
        if preds is not None:
            print(f"Found {len(preds)} face(s)")
            for i, landmarks in enumerate(preds):
                print(f"Face {i+1}: {landmarks.shape[0]} landmarks detected")
            return preds
        else:
            print("No faces detected")
            return None
    
    def process_face(self, image_path: str, output_size: Tuple[int, int] = (256, 256)) -> Optional[np.ndarray]:
        """
        Complete face processing pipeline: load, detect, align, and crop face.
        Handles both regular face images and document images.
        
        Args:
            image_path (str): Path to input image
            output_size (Tuple[int, int]): Desired output size
            
        Returns:
            Optional[np.ndarray]: Processed face image or None if processing fails
            
        Raises:
            ValueError: If no human face is detected in document images
        """
        try:
            # Load image
            image = self.load_image(image_path)
            print(f"Loaded image shape: {image.shape}")
            
            # Check if this is a document image
            is_document = self.is_document_image(image)
            
            if is_document:
                print("Detected document image (passport, driving license, etc.)")
                
                # Extract face from document
                face_region = self.extract_face_from_document(image)
                
                if face_region is None:
                    raise ValueError("No human face detected in the document image. Please provide an image with a clear human face.")
                
                print(f"Successfully extracted face region from document: {face_region.shape}")
                
                # Process the extracted face region
                image = face_region
            else:
                print("Processing as regular face image")
            
            # Detect faces in the image (original or extracted face region)
            landmarks_list = self.detect_faces(image)
            if not landmarks_list:
                if is_document:
                    raise ValueError("No human face detected in the document image. Please provide an image with a clear human face.")
                else:
                    print("No faces detected in image")
                    return None
            
            # Use first detected face
            landmarks = landmarks_list[0]
            
            # Auto-upright the image
            upright_image, upright_landmarks = self.auto_upright_image(image)
            
            # Align and crop face
            aligned_face, _ = self.align_face(upright_image, upright_landmarks, output_size)
            
            return aligned_face
            
        except Exception as e:
            print(f"Error processing face: {e}")
            return None
    
    def save_image(self, image: np.ndarray, output_path: str) -> bool:
        """
        Save image to file.
        
        Args:
            image (np.ndarray): Image to save
            output_path (str): Output file path
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            # Convert RGB to BGR for OpenCV
            if len(image.shape) == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            cv2.imwrite(output_path, image_bgr)
            print(f"Image saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error saving image: {e}")
            return False


def main():
    """Example usage of the FacePreprocessor class."""
    # Initialize preprocessor
    preprocessor = FacePreprocessor(device='cpu')
    
    # Example image paths - you can test with both regular face images and document images
    image_paths = [
        "/Users/ravee/Downloads/VGGFace2Kaggle/train/n000261/0037_01.jpg",  # Regular face image
        # Add your document image paths here for testing
        # "path/to/passport_image.jpg",
        # "path/to/driving_license_image.jpg"
    ]
    
    for image_path in image_paths:
        print(f"\n{'='*50}")
        print(f"Processing: {image_path}")
        print(f"{'='*50}")
        
        try:
            # Process face
            aligned_face = preprocessor.process_face(image_path)
            
            if aligned_face is not None:
                # Save processed face
                output_path = f"aligned_faces/aligned_face_{os.path.basename(image_path)}"
                preprocessor.save_image(aligned_face, output_path)
                
                # Display results
                plt.figure(figsize=(12, 4))
                
                # Original image
                original = preprocessor.load_image(image_path)
                plt.subplot(1, 3, 1)
                plt.title("Original Image")
                plt.imshow(original)
                plt.axis("off")
                
                # Aligned face
                plt.subplot(1, 3, 2)
                plt.title("Aligned Face")
                plt.imshow(aligned_face)
                plt.axis("off")
                
                plt.tight_layout()
                plt.show()
            else:
                print("Failed to process face")
                
        except ValueError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
