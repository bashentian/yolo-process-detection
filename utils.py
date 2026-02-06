import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime


def create_video_from_images(image_dir: str, output_path: str, 
                              fps: int = 30, pattern: str = "*.jpg"):
    images = sorted(Path(image_dir).glob(pattern))
    
    if not images:
        raise ValueError(f"No images found in {image_dir}")
    
    first_image = cv2.imread(str(images[0]))
    height, width = first_image.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for image_path in images:
        frame = cv2.imread(str(image_path))
        writer.write(frame)
    
    writer.release()
    print(f"Video created from {len(images)} images: {output_path}")


def split_video(video_path: str, output_dir: str, 
                segment_duration: int = 30):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    segment_frames = int(fps * segment_duration)
    num_segments = total_frames // segment_frames
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for seg_idx in range(num_segments):
        output_file = output_path / f"segment_{seg_idx:03d}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))
        
        start_frame = seg_idx * segment_frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for _ in range(segment_frames):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
        
        writer.release()
        print(f"Created segment {seg_idx + 1}/{num_segments}")
    
    cap.release()


def resize_video(input_path: str, output_path: str, 
                target_width: int = 640, target_height: int = 480):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        resized = cv2.resize(frame, (target_width, target_height))
        writer.write(resized)
    
    cap.release()
    writer.release()
    print(f"Video resized to {target_width}x{target_height}: {output_path}")


def extract_audio_from_video(video_path: str, output_path: str):
    try:
        import moviepy.editor as mp
        
        video = mp.VideoFileClip(video_path)
        audio = video.audio
        
        if audio is not None:
            audio.write_audiofile(output_path)
            print(f"Audio extracted to: {output_path}")
        else:
            print("No audio track found in the video")
    except ImportError:
        print("moviepy not installed. Install with: pip install moviepy")


def merge_video_audio(video_path: str, audio_path: str, output_path: str):
    try:
        import moviepy.editor as mp
        
        video = mp.VideoFileClip(video_path)
        audio = mp.AudioFileClip(audio_path)
        
        final_video = video.set_audio(audio)
        final_video.write_videofile(output_path)
        
        print(f"Video and audio merged: {output_path}")
    except ImportError:
        print("moviepy not installed. Install with: pip install moviepy")


def create_comparison_video(original_path: str, annotated_path: str, 
                           output_path: str):
    cap_orig = cv2.VideoCapture(original_path)
    cap_annot = cv2.VideoCapture(annotated_path)
    
    fps = cap_orig.get(cv2.CAP_PROP_FPS)
    width = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_width = width * 2
    output_height = height
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
    
    while True:
        ret1, frame1 = cap_orig.read()
        ret2, frame2 = cap_annot.read()
        
        if not ret1 or not ret2:
            break
        
        combined = np.hstack([frame1, frame2])
        writer.write(combined)
    
    cap_orig.release()
    cap_annot.release()
    writer.release()
    
    print(f"Comparison video created: {output_path}")


def add_timestamp(frame: np.ndarray, timestamp: str, 
                  position: Tuple[int, int] = (10, 30),
                  font_scale: float = 1.0, color: Tuple[int, int, int] = (255, 255, 255)):
    cv2.putText(frame, timestamp, position, 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
    return frame


def add_watermark(frame: np.ndarray, watermark_text: str,
                 position: str = "bottom-right",
                 alpha: float = 0.5):
    overlay = frame.copy()
    
    h, w = frame.shape[:2]
    font_scale = min(w, h) / 1000
    
    (text_width, text_height), _ = cv2.getTextSize(
        watermark_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
    )
    
    if position == "bottom-right":
        x = w - text_width - 20
        y = h - 20
    elif position == "bottom-left":
        x = 20
        y = h - 20
    elif position == "top-right":
        x = w - text_width - 20
        y = text_height + 20
    else:  # top-left
        x = 20
        y = text_height + 20
    
    cv2.putText(overlay, watermark_text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
    
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def create_video_summary(video_path: str, output_path: str, 
                         summary_duration: int = 30,
                         num_frames: int = 30):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_interval = total_frames // num_frames
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frames_per_summary_frame = int(fps * summary_duration / num_frames)
    
    for i in range(num_frames):
        frame_num = i * frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if ret:
            for _ in range(frames_per_summary_frame):
                writer.write(frame)
    
    cap.release()
    writer.release()
    
    print(f"Video summary created: {output_path} ({summary_duration}s)")


def detect_video_properties(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    
    properties = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    }
    
    cap.release()
    
    return properties


def create_dataset_from_video(video_path: str, output_dir: str,
                             frame_interval: int = 30,
                             resize: Optional[Tuple[int, int]] = None):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            if resize:
                frame = cv2.resize(frame, resize)
            
            output_file = output_path / f"frame_{saved_count:06d}.jpg"
            cv2.imwrite(str(output_file), frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    
    print(f"Extracted {saved_count} frames from {total_frames} total frames")
    print(f"Images saved to: {output_dir}")


def batch_resize_images(input_dir: str, output_dir: str,
                        target_size: Tuple[int, int] = (640, 640)):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
    
    for i, img_path in enumerate(image_files):
        img = cv2.imread(str(img_path))
        resized = cv2.resize(img, target_size)
        output_file = output_path / img_path.name
        cv2.imwrite(str(output_file), resized)
        
        print(f"\rResized {i+1}/{len(image_files)} images", end="")
    
    print(f"\nResized {len(image_files)} images to {target_size}")


if __name__ == "__main__":
    print("Video Utilities Module")
    print("Import this module to use utility functions for video processing")
