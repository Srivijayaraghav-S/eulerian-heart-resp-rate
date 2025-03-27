import numpy as np
from scipy import signal

# Calculate heart rate from FFT peaks
def find_heart_rate(fft, freqs, freq_min, freq_max):
    fft_maximums = []

    for i in range(fft.shape[0]):
        if freq_min <= freqs[i] <= freq_max:
            fftMap = abs(fft[i])
            fft_maximums.append(fftMap.max())
        else:
            fft_maximums.append(0)

    peaks, properties = signal.find_peaks(fft_maximums)
    max_peak = -1
    max_freq = 0

    # Find frequency with max amplitude in peaks
    for peak in peaks:
        if fft_maximums[peak] > max_freq:
            max_freq = fft_maximums[peak]
            max_peak = peak
            
    print("[DEBUG] Heart rate frequency peak:", freqs[max_peak], "Hz")
    print("[DEBUG] Heart rate BPM:", freqs[max_peak] * 60)
    return freqs[max_peak] * 60

def find_respiration_rate(fft_values, freqs):
    respiration_range = (0.1, 0.5)  # Hz (6-30 breaths per minute)
    valid_indices = np.where((freqs >= respiration_range[0]) & (freqs <= respiration_range[1]))[0]
    
    respiration_fft_values = fft_values[valid_indices]
    respiration_freqs = freqs[valid_indices]
    
    if len(respiration_fft_values) == 0:
        return 0.0  # No respiration detected
    
    max_peak = np.argmax(respiration_fft_values)
    respiration_rate_hz = respiration_freqs[max_peak]
    respiration_rate_bpm = respiration_rate_hz * 60  # Convert Hz to breaths per minute
    
    print("[DEBUG] Respiration FFT Peaks:", respiration_fft_values)
    print("[DEBUG] Respiration rate frequency peak:", respiration_rate_hz, "Hz")
    print("[DEBUG] Respiration rate BPM:", respiration_rate_bpm)
    
    return respiration_rate_bpm

def find_hrv(rr_intervals):
    if len(rr_intervals) < 2:
        print("[DEBUG] Not enough RR intervals for HRV computation.")
        return 0
    
    hrv_sdnn = np.std(rr_intervals)  # Standard deviation of RR intervals
    
    print("[DEBUG] RR Intervals:", rr_intervals)
    print("[DEBUG] HRV (SDNN):", hrv_sdnn, "ms")
    
    return hrv_sdnn
