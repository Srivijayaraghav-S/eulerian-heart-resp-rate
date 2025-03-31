import cv2
import pyramids
import heartrate
import preprocessing
import eulerian
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# Frequency ranges
HR_FREQ_MIN = 1.0  # ~60 bpm
HR_FREQ_MAX = 2.0   # ~120 bpm
RESP_FREQ_MIN = 0.1  # ~6 bpm
RESP_FREQ_MAX = 0.5   # ~30 bpm

def main():
    # Preprocessing
    print("Reading + preprocessing video...")
    try:
        video_frames, frame_ct, fps = preprocessing.read_video("videos/collab_cam.mov")
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
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Physiological Signal Analysis', fontsize=16)

    # Initialize data containers
    time_points = np.arange(frame_ct) / fps
    raw_signal = np.zeros(frame_ct)
    heart_rates = np.zeros(frame_ct)
    respiration_rates = np.zeros(frame_ct)
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

    # Plot 2: Frequency spectrum
    freq_line, = ax2.plot([], [], 'r-', linewidth=1, label='FFT Magnitude')
    hr_marker = ax2.axvline(0, color='g', linestyle='--', label='HR Peak', visible=False)
    resp_marker = ax2.axvline(0, color='m', linestyle='--', label='Respiration Peak', visible=False)
    ax2.set_title('Frequency Spectrum')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude')
    ax2.set_xlim(0, 2.5)
    ax2.grid(True)
    ax2.legend()

    # Plot 3: Extracted rates
    hr_line, = ax3.plot([], [], 'g-', label='Heart Rate')
    resp_line, = ax3.plot([], [], 'm-', label='Respiration Rate')
    ax3.set_title('Physiological Rates')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Rate (BPM)')
    ax3.set_xlim(0, frame_ct/fps)
    ax3.set_ylim(0, 150)
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
            
            # Update frequency plot
            if len(freqs) > 0:
                freq_line.set_data(freqs, magnitudes)
                ax2.relim()
                ax2.autoscale_view()
            
            # Find heart rate
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
                    
                    # Update HR marker
                    hr_freq = smoothed_hr / 60
                    hr_marker.set_xdata([hr_freq, hr_freq])
                    hr_marker.set_visible(True)
            
            # Find respiration rate
            resp_mask = (freqs >= RESP_FREQ_MIN) & (freqs <= RESP_FREQ_MAX)
            if np.any(resp_mask):
                resp_rate = heartrate.find_respiration_rate(
                    magnitudes[resp_mask],
                    freqs[resp_mask]
                )
                if resp_rate > 0:
                    respiration_rates[i] = resp_rate
                    # Update respiration marker
                    resp_freq = resp_rate / 60
                    resp_marker.set_xdata([resp_freq, resp_freq])
                    resp_marker.set_visible(True)
            
            # Update rates plot
            hr_line.set_data(time_points[:i], heart_rates[:i])
            resp_line.set_data(time_points[:i], respiration_rates[:i])
            ax3.relim()
            ax3.autoscale_view()
            
            plt.pause(0.001)

    # Apply Eulerian magnification to the video pyramid
    print("Applying Eulerian magnification...")
    for i in range(1, len(lap_video)-1):  # Skip first and last levels
        # Apply bandpass filter to the video level
        result, fft, frequencies = eulerian.fft_filter(lap_video[i], HR_FREQ_MIN, HR_FREQ_MAX, fps)
        lap_video[i] += result * 50  # Amplify the signal

    # Calculate statistics
    valid_hr = heart_rates[heart_rates > 0]
    valid_resp = respiration_rates[respiration_rates > 0]

    if len(valid_hr) > 0:
        avg_hr = np.mean(valid_hr)
        print(f"\nFinal Heart Rate: {avg_hr:.1f} ± {np.std(valid_hr):.1f} bpm")
        print(f"All valid readings: {valid_hr}")
        
    if len(valid_resp) > 0:
        print(f"\nFinal Respiration Rate: {np.mean(valid_resp):.1f} ± {np.std(valid_resp):.1f} breaths/min")

    plt.show()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()