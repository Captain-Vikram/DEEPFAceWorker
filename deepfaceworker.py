import cv2
import numpy as np
import tensorflow as tf
from deepface import DeepFace
from mtcnn import MTCNN
import multiprocessing
import time
from moviepy import VideoFileClip
import os
from tqdm import tqdm  # For better progress tracking

MAX_CORES = 14
num_cores = min(multiprocessing.cpu_count(), MAX_CORES)
print(f"Using {num_cores} CPU cores out of {multiprocessing.cpu_count()} available cores")

# Configure TensorFlow to use limited CPU cores
tf.config.threading.set_intra_op_parallelism_threads(num_cores)
tf.config.threading.set_inter_op_parallelism_threads(num_cores)

# Optimize OpenCV for multi-threading with limited cores
cv2.setNumThreads(num_cores)

# Input and output video paths
video_path = "input_video.mp4"
temp_output_path = "temp_output.mp4"
final_output_path = "output_video.mp4"
temp_dir = "temp_frames"

# Create temporary directory if it doesn't exist
os.makedirs(temp_dir, exist_ok=True)

def extract_frames(video_path, output_dir):
    """Extract all frames from video to disk for parallel processing"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"Extracting {total_frames} frames from video...")
    for i in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{output_dir}/frame_{i:06d}.jpg", frame)
    
    cap.release()
    return total_frames, frame_width, frame_height, fps

def process_frame(frame_path):
    """Process a single frame with emotion detection"""
    # Load frame
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Error loading {frame_path}")
        return frame_path, None
    
    # Initialize MTCNN for each process to avoid conflicts
    detector = MTCNN()
    
    # Detect faces
    faces = detector.detect_faces(frame)
    
    if faces:
        # Find the largest face
        largest_face = max(faces, key=lambda face: face['box'][2] * face['box'][3])
        x, y, w, h = largest_face['box']
        
        # Extract face region
        face_img = frame[y:y+h, x:x+w]
        temp_face_path = frame_path.replace('.jpg', '_face.jpg')
        cv2.imwrite(temp_face_path, face_img)
        
        try:
            # Perform emotion analysis
            result = DeepFace.analyze(temp_face_path, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            
            # Draw bounding box and annotate emotion
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Emotion: {emotion}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Save the processed frame
            cv2.imwrite(frame_path.replace('.jpg', '_processed.jpg'), frame)
            
            # Clean up
            os.remove(temp_face_path)
        except Exception as e:
            print(f"Error processing {frame_path}: {e}")
    
    return frame_path, frame

def main():
    start_time = time.time()
    
    # Extract all frames from video
    total_frames, frame_width, frame_height, fps = extract_frames(video_path, temp_dir)
    print(f"Extraction complete. Processing {total_frames} frames with emotion detection...")
    
    # Create a list of all frame paths
    frame_paths = [f"{temp_dir}/frame_{i:06d}.jpg" for i in range(total_frames)]
    
    # Process frames in parallel
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = list(tqdm(pool.imap(process_frame, frame_paths), total=len(frame_paths)))
    
    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (frame_width, frame_height))
    
    print("Creating output video...")
    for i in tqdm(range(total_frames)):
        processed_path = f"{temp_dir}/frame_{i:06d}_processed.jpg"
        if os.path.exists(processed_path):
            frame = cv2.imread(processed_path)
        else:
            frame = cv2.imread(f"{temp_dir}/frame_{i:06d}.jpg")
        out.write(frame)
    
    out.release()
    
    # Add audio back to the video
    print("Adding audio to the video...")
    try:
        original_video = VideoFileClip(video_path)
        original_audio = original_video.audio
        
        if original_audio is not None:
            processed_video = VideoFileClip(temp_output_path)
            final_video = processed_video.with_audio(original_audio)
            final_video.write_videofile(final_output_path, 
                                      codec="libx264", 
                                      audio_codec="aac",
                                      threads=num_cores)
            processed_video.close()
            final_video.close()
            print(f"✅ Final video with audio saved to {final_output_path}")
        else:
            print("⚠️ No audio found in the original video")
            # Just rename the temp output to final output
            os.rename(temp_output_path, final_output_path)
        
        original_video.close()
    except Exception as e:
        print(f"Error adding audio: {e}")
        print("Using video without audio as final output")
        os.rename(temp_output_path, final_output_path)
    
    # Clean up temporary files
    print("Cleaning up temporary files...")
    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))
    os.rmdir(temp_dir)
    
    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Processing speed: {total_frames / total_time:.2f} fps")

if __name__ == "__main__":
    main()