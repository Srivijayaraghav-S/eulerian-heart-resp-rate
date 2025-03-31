import cv2
import numpy as np

def build_gaussian_pyramid(img, levels):
    """Build Gaussian pyramid ensuring consistent dimensions"""
    pyramid = [img.astype("float32")]
    
    for _ in range(levels-1):
        # Get current dimensions
        h, w = pyramid[-1].shape[:2]
        # Ensure even dimensions before pyrDown
        if h % 2 != 0:
            pyramid[-1] = pyramid[-1][:-1]
        if w % 2 != 0:
            pyramid[-1] = pyramid[-1][:, :-1]
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    
    return pyramid

def build_laplacian_pyramid(img, levels):
    """Build Laplacian pyramid with consistent dimensions"""
    gaussian = build_gaussian_pyramid(img, levels)
    laplacian = []
    
    for i in range(levels-1):
        # Get dimensions before upsampling
        h, w = gaussian[i+1].shape[:2]
        upsampled = cv2.pyrUp(gaussian[i+1], dstsize=(w*2, h*2))
        
        # Ensure dimensions match
        h_up, w_up = upsampled.shape[:2]
        h_curr, w_curr = gaussian[i].shape[:2]
        
        # Crop if necessary
        if h_up != h_curr or w_up != w_curr:
            min_h = min(h_up, h_curr)
            min_w = min(w_up, w_curr)
            upsampled = upsampled[:min_h, :min_w]
            gaussian[i] = gaussian[i][:min_h, :min_w]
        
        laplacian.append(gaussian[i] - upsampled)
    
    laplacian.append(gaussian[-1])
    return laplacian

def build_video_pyramid(frames):
    """Build video pyramid with consistent dimensions"""
    if not frames:
        return []
    
    # Get dimensions from first frame
    first_frame = frames[0]
    levels = 3
    pyramid_sample = build_laplacian_pyramid(first_frame, levels)
    
    # Initialize pyramid with correct dimensions
    lap_video = [
        np.zeros((len(frames), *pyramid_sample[i].shape), dtype="float32")
        for i in range(levels)
    ]
    
    # Build pyramid for each frame
    for i, frame in enumerate(frames):
        pyramid = build_laplacian_pyramid(frame, levels)
        for level in range(levels):
            # Ensure dimensions match before assignment
            h_pyr, w_pyr = pyramid[level].shape[:2]
            h_dest, w_dest = lap_video[level][i].shape[:2]
            
            if h_pyr == h_dest and w_pyr == w_dest:
                lap_video[level][i] = pyramid[level]
            else:
                # Resize if dimensions don't match
                lap_video[level][i] = cv2.resize(pyramid[level], (w_dest, h_dest))
    
    return lap_video

def collapse_laplacian_video_pyramid(video, frame_ct):
    """Collapse pyramid with dimension checks"""
    collapsed = []
    
    for i in range(frame_ct):
        if i >= len(video[-1]):
            continue
            
        current = video[-1][i]
        
        for level in range(len(video)-2, -1, -1):
            if i >= len(video[level]):
                continue
                
            h, w = current.shape[:2]
            upsampled = cv2.pyrUp(current, dstsize=(w*2, h*2))
            
            # Ensure dimensions match
            h_up, w_up = upsampled.shape[:2]
            h_level, w_level = video[level][i].shape[:2]
            
            if h_up != h_level or w_up != w_level:
                min_h = min(h_up, h_level)
                min_w = min(w_up, w_level)
                upsampled = upsampled[:min_h, :min_w]
                level_frame = video[level][i][:min_h, :min_w]
            else:
                level_frame = video[level][i]
            
            current = upsampled + level_frame
        
        # Normalize and convert
        current = np.clip(current, 0, 1)
        collapsed.append((current * 255).astype("uint8"))
    
    return collapsed