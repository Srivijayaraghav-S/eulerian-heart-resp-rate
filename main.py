import cv2
import pyramids
import heartrate
import preprocessing
import eulerian
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Frequency ranges
HR_FREQ_MIN = 1.0  # ~60 bpm
HR_FREQ_MAX = 2.0   # ~120 bpm
RESP_FREQ_MIN = 0.1  # ~6 bpm
RESP_FREQ_MAX = 0.5   # ~30 bpm

# Preprocessing
print("Reading + preprocessing video...")
video_frames, frame_ct, fps = preprocessing.read_video("videos/rohin_active.mov")

# Build pyramid
print("Building Laplacian video pyramid...")
lap_video = pyramids.build_video_pyramid(video_frames)

# Set up plots
plt.style.use('ggplot')
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
fig.suptitle('Physiological Signal Analysis', fontsize=16)

# Initialize data containers
time_points = np.arange(frame_ct) / fps
raw_signal = np.zeros(frame_ct)
heart_rates = np.zeros(frame_ct)
respiration_rates = np.zeros(frame_ct)

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

# Process video frames and apply magnification
amplified_video = []
for i in range(frame_ct):
    if i >= len(lap_video[0]):  # Ensure we don't exceed frame count
        continue
        
    # Get current frame from all pyramid levels
    current_frames = [level[i] for level in lap_video]
    
    # Store raw signal (mean of middle pyramid level)
    if len(lap_video) > 1:
        raw_signal[i] = np.mean(lap_video[1][i])  # Use middle pyramid level
        
    # Update raw signal plot
    if i > 1:
        raw_line.set_data(time_points[:i], raw_signal[:i])
        ax1.relim()
        ax1.autoscale_view()
        
        # Compute FFT periodically
        if i % fps == 0 and i > fps:
            # Use last 2 seconds of data
            fft_size = 2 * fps
            start_idx = max(0, i - fft_size)
            segment = raw_signal[start_idx:i]
            
            if len(segment) < 10:
                continue
                
            # Apply windowing and compute FFT
            window = np.hanning(len(segment))
            windowed = segment * window
            fft = np.fft.rfft(windowed)
            freqs = np.fft.rfftfreq(len(windowed), d=1.0/fps)
            magnitudes = np.abs(fft)
            
            # Update frequency plot
            if len(freqs) > 0:
                freq_line.set_data(freqs, magnitudes)
                ax2.relim()
                ax2.autoscale_view()
                
                # Find heart rate peak
                hr_mask = (freqs >= HR_FREQ_MIN) & (freqs <= HR_FREQ_MAX)
                if np.any(hr_mask):
                    hr_peak_idx = np.argmax(magnitudes[hr_mask])
                    hr_freq = freqs[hr_mask][hr_peak_idx]
                    hr_marker.set_xdata([hr_freq, hr_freq])
                    hr_marker.set_visible(True)
                    heart_rates[i] = hr_freq * 60
                
                # Find respiration peak
                resp_mask = (freqs >= RESP_FREQ_MIN) & (freqs <= RESP_FREQ_MAX)
                if np.any(resp_mask):
                    resp_peak_idx = np.argmax(magnitudes[resp_mask])
                    resp_freq = freqs[resp_mask][resp_peak_idx]
                    resp_marker.set_xdata([resp_freq, resp_freq])
                    resp_marker.set_visible(True)
                    respiration_rates[i] = resp_freq * 60
                
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

# Reconstruct the final video
print("Rebuilding final video...")
amplified_frames = pyramids.collapse_laplacian_video_pyramid(lap_video, frame_ct)

# Calculate statistics
valid_hr = heart_rates[heart_rates > 0]
valid_resp = respiration_rates[respiration_rates > 0]

if len(valid_hr) > 0:
    print(f"Heart rate: {np.mean(valid_hr) + np.std(valid_hr):.1f} bpm")
    print("Valid Heart Rate Array: ", valid_hr)
if len(valid_resp) > 0:
    print(f"Respiration rate: {np.mean(valid_resp) + np.std(valid_resp):.1f} breaths/min")
    print("Valid Respiration Rate Array: ", valid_resp)

plt.show()

# Display amplified video
print("Displaying final video...")
for frame in amplified_frames:
    frame = cv2.convertScaleAbs(frame)
    cv2.imshow("Amplified Physiological Signals", frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()