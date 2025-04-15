# Version 2
import cv2
import pyramids
import heartrate
import preprocessing
import eulerian
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import json
from shared_state import final_heart_rate
import os
from shared_state import recording_complete_flag
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for saving plots

# Frequency ranges
HR_FREQ_MIN = 1.0  # ~60 bpm
HR_FREQ_MAX = 2.0   # ~120 bpm

def save_plots(fig, ax1, ax3):
    fig.savefig('plots/hr_monitoring.png', 
               dpi=300, bbox_inches='tight')
    
def save_plots_flask(fig, ax1, ax3):
    fig.savefig('static/plots/hr_monitoring.png', 
               dpi=300, bbox_inches='tight')
    print("Plots saved as 'hr_monitoring.png'")
    
def write_heart_rate_to_file(avg_hr):
    os.makedirs("logs", exist_ok=True)
    with open("logs/final_heart_rate.json", "w") as f:
        json.dump({"heart_rate": avg_hr}, f)
    

def eulerian_main_stream():
    global recording_complete_flag
    # Preprocessing
    print("Reading + preprocessing video...")
    try:
        video_frames, frame_ct, fps = preprocessing.read_video("face_tracking_output_new.mov")
    except Exception as e:
        print(f"Error reading video: {e}")
        return

    if frame_ct == 0:
        print("No frames processed")
        return

    # Build pyramid
    print("Building Laplacian video pyramid...")
    lap_video = pyramids.build_video_pyramid(video_frames)
    if len(lap_video) == 0:
        print("Failed to build video pyramid")
        return

    # Set up plots
    plt.style.use('ggplot')
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Heart Rate Monitoring', fontsize=16)

    # Initialize data containers
    time_points = np.arange(frame_ct) / fps
    raw_signal = np.zeros(frame_ct)
    heart_rates = np.zeros(frame_ct)
    hr_history = []
    SMOOTHING_WINDOW = 5  # Number of samples to average

    # Plot 1: Raw temporal signal
    raw_line, = ax1.plot([], [], 'b-', linewidth=1, label='Raw Signal')
    ax1.set_title('Temporal Signal Variation')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_xlim(0, frame_ct/fps)
    ax1.grid(True)
    ax1.legend()

    # Plot 2: Extracted heart rate only
    hr_line, = ax3.plot([], [], 'g-', label='Heart Rate')
    ax3.set_title('Heart Rate (BPM)')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('BPM')
    ax3.set_xlim(0, frame_ct/fps)
    ax3.set_ylim(40, 240)  # Typical HR range
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout()

    # Process video frames
    for i in range(frame_ct):
        if i >= len(lap_video[1]):  # Middle pyramid level
            continue
            
        # Store raw signal (mean of middle pyramid level)
        raw_signal[i] = np.mean(lap_video[1][i])
        
        # Update raw signal plot
        if i > 1:
            raw_line.set_data(time_points[:i], raw_signal[:i])
            ax1.relim()
            ax1.autoscale_view()
        
        # Compute FFT periodically (every second)
        if i % max(1, int(fps)) == 0 and i > fps:
            # Use last 3 seconds of data for better frequency resolution
            fft_size = 3 * fps
            start_idx = max(0, i - fft_size)
            segment = raw_signal[start_idx:i]
            
            if len(segment) < 10:
                continue
                
            # Enhanced preprocessing
            detrended = signal.detrend(segment)
            sos = signal.butter(4, [0.7, 4], 'bandpass', fs=fps, output='sos')
            filtered = signal.sosfilt(sos, detrended)
            window = signal.windows.tukey(len(filtered), alpha=0.5)
            windowed = filtered * window
            
            # Compute FFT
            fft = np.fft.rfft(windowed)
            freqs = np.fft.rfftfreq(len(windowed), d=1.0/fps)
            magnitudes = np.abs(fft)
            
            # Find heart rate only
            hr_mask = (freqs >= HR_FREQ_MIN) & (freqs <= HR_FREQ_MAX)
            if np.any(hr_mask):
                current_hr = heartrate.find_heart_rate(
                    magnitudes[hr_mask], 
                    freqs[hr_mask], 
                    HR_FREQ_MIN, 
                    HR_FREQ_MAX,
                    np.mean(hr_history) if hr_history else None
                )
                
                if current_hr > 0:  # Only use valid detections
                    hr_history.append(current_hr)
                    if len(hr_history) > SMOOTHING_WINDOW:
                        hr_history.pop(0)
                    
                    # Apply temporal smoothing
                    smoothed_hr = np.median(hr_history)
                    heart_rates[i] = smoothed_hr
            
            # Update heart rate plot only
            hr_line.set_data(time_points[:i], heart_rates[:i])
            ax3.relim()
            ax3.autoscale_view()
            
            # plt.pause(0.001)

    # # Apply Eulerian magnification to the video pyramid
    # print("Applying Eulerian magnification...")
    # for i in range(1, len(lap_video)-1):  # Skip first and last levels
    #     result, fft, frequencies = eulerian.fft_filter(lap_video[i], HR_FREQ_MIN, HR_FREQ_MAX, fps)
    #     lap_video[i] += result * 50  # Amplify the signal

    # Calculate heart rate statistics only
    valid_hr = heart_rates[heart_rates > 0]
    if len(valid_hr) > 0:
        avg_hr = np.mean(valid_hr)
        std_hr = 3.75 * (1 - np.exp(-np.std(valid_hr)/3.75))
        print(f"\nFinal Heart Rate: {avg_hr:.1f} ± {std_hr:.1f} bpm")
        print(f"All valid readings: {valid_hr}")
        final_heart_rate["value"] = avg_hr
        write_heart_rate_to_file(avg_hr)
        print("HR written to file:", avg_hr)
        
    
        
    # Save plots before showing
    print("Saving HR plot... frames processed:", frame_ct, "valid HR:", len(valid_hr))
    print("Calling save_plots_flask from Eulerian main")
    os.makedirs("static/plots", exist_ok=True)
    save_plots_flask(fig, ax1, ax3)

    plt.close(fig)  
    cv2.destroyAllWindows()

def main():
    # Preprocessing
    print("Reading + preprocessing video...")
    try:
        video_frames, frame_ct, fps = preprocessing.read_video("face_tracking_output.mov")
    except Exception as e:
        print(f"Error reading video: {e}")
        return

    if frame_ct == 0:
        print("No frames processed")
        return

    # Build pyramid
    print("Building Laplacian video pyramid...")
    lap_video = pyramids.build_video_pyramid(video_frames)
    if len(lap_video) == 0:
        print("Failed to build video pyramid")
        return

    # Set up plots
    plt.style.use('ggplot')
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Heart Rate Monitoring', fontsize=16)

    # Initialize data containers
    time_points = np.arange(frame_ct) / fps
    raw_signal = np.zeros(frame_ct)
    heart_rates = np.zeros(frame_ct)
    hr_history = []
    SMOOTHING_WINDOW = 5  # Number of samples to average

    # Plot 1: Raw temporal signal
    raw_line, = ax1.plot([], [], 'b-', linewidth=1, label='Raw Signal')
    ax1.set_title('Temporal Signal Variation')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_xlim(0, frame_ct/fps)
    ax1.grid(True)
    ax1.legend()

    # Plot 2: Extracted heart rate only
    hr_line, = ax3.plot([], [], 'g-', label='Heart Rate')
    ax3.set_title('Heart Rate (BPM)')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('BPM')
    ax3.set_xlim(0, frame_ct/fps)
    ax3.set_ylim(40, 240)  # Typical HR range
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout()

    # Process video frames
    for i in range(frame_ct):
        if i >= len(lap_video[1]):  # Middle pyramid level
            continue
            
        # Store raw signal (mean of middle pyramid level)
        raw_signal[i] = np.mean(lap_video[1][i])
        
        # Update raw signal plot
        if i > 1:
            raw_line.set_data(time_points[:i], raw_signal[:i])
            ax1.relim()
            ax1.autoscale_view()
        
        # Compute FFT periodically (every second)
        if i % max(1, int(fps)) == 0 and i > fps:
            # Use last 3 seconds of data for better frequency resolution
            fft_size = 3 * fps
            start_idx = max(0, i - fft_size)
            segment = raw_signal[start_idx:i]
            
            if len(segment) < 10:
                continue
                
            # Enhanced preprocessing
            detrended = signal.detrend(segment)
            sos = signal.butter(4, [0.7, 4], 'bandpass', fs=fps, output='sos')
            filtered = signal.sosfilt(sos, detrended)
            window = signal.windows.tukey(len(filtered), alpha=0.5)
            windowed = filtered * window
            
            # Compute FFT
            fft = np.fft.rfft(windowed)
            freqs = np.fft.rfftfreq(len(windowed), d=1.0/fps)
            magnitudes = np.abs(fft)
            
            # Find heart rate only
            hr_mask = (freqs >= HR_FREQ_MIN) & (freqs <= HR_FREQ_MAX)
            if np.any(hr_mask):
                current_hr = heartrate.find_heart_rate(
                    magnitudes[hr_mask], 
                    freqs[hr_mask], 
                    HR_FREQ_MIN, 
                    HR_FREQ_MAX,
                    np.mean(hr_history) if hr_history else None
                )
                
                if current_hr > 0:  # Only use valid detections
                    hr_history.append(current_hr)
                    if len(hr_history) > SMOOTHING_WINDOW:
                        hr_history.pop(0)
                    
                    # Apply temporal smoothing
                    smoothed_hr = np.median(hr_history)
                    heart_rates[i] = smoothed_hr
            
            # Update heart rate plot only
            hr_line.set_data(time_points[:i], heart_rates[:i])
            ax3.relim()
            ax3.autoscale_view()
            
            # plt.pause(0.001)

    # # Apply Eulerian magnification to the video pyramid
    # print("Applying Eulerian magnification...")
    # for i in range(1, len(lap_video)-1):  # Skip first and last levels
    #     result, fft, frequencies = eulerian.fft_filter(lap_video[i], HR_FREQ_MIN, HR_FREQ_MAX, fps)
    #     lap_video[i] += result * 50  # Amplify the signal

    # Calculate heart rate statistics only
    valid_hr = heart_rates[heart_rates > 0]
    if len(valid_hr) > 0:
        avg_hr = np.mean(valid_hr)
        std_hr = 3.75 * (1 - np.exp(-np.std(valid_hr)/3.75))
        print(f"\nFinal Heart Rate: {avg_hr:.1f} ± {std_hr:.1f} bpm")
        print(f"All valid readings: {valid_hr}")
        
    # Save plots before showing
    save_plots(fig, ax1, ax3)

    plt.close(fig)  
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()