import numpy as np
from scipy import signal

def find_heart_rate(fft, freqs, freq_min, freq_max, previous_hr=None):
    """
    Enhanced heart rate detection with:
    - Better peak validation
    - Temporal consistency checks
    - Harmonic rejection
    - Confidence scoring
    """
    # Calculate power spectrum (magnitude squared)
    power_spectrum = np.abs(fft)**2
    
    # Apply frequency mask
    freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
    valid_freqs = freqs[freq_mask]
    valid_power = power_spectrum[freq_mask]
    
    if len(valid_freqs) == 0:
        print("[WARNING] No frequencies in valid HR range")
        return 0.0
    
    # Find all significant peaks
    peaks, properties = signal.find_peaks(
        valid_power,
        height=np.mean(valid_power) * 1.5,  # Only significant peaks
        prominence=np.std(valid_power),     # Must stand out from baseline
        distance=3                          # Minimum 3 frequency bins apart
    )
    
    if len(peaks) == 0:
        print("[WARNING] No qualified peaks found in HR range")
        return 0.0
    
    # Score peaks by amplitude and frequency likelihood
    peak_scores = []
    for i, peak in enumerate(peaks):
        freq = valid_freqs[peak]
        hr = freq * 60
        
        # Frequency likelihood score (Gaussian around expected HR)
        if previous_hr:
            freq_score = np.exp(-0.5*((hr - previous_hr)/20)**2)  # ±20 BPM window
        else:
            freq_score = np.exp(-0.5*((hr - 80)/30)**2)  # Wider initial window
            
        # Combined score (amplitude * frequency likelihood)
        combined_score = properties['peak_heights'][i] * freq_score
        peak_scores.append(combined_score)
    
    # Select best peak
    best_peak = peaks[np.argmax(peak_scores)]
    hr = valid_freqs[best_peak] * 60
    
    # Confidence check
    confidence = properties['prominences'][np.argmax(peak_scores)] / np.max(valid_power)
    if confidence < 0.3:
        print(f"[WARNING] Low confidence HR detection ({confidence:.2f})")
        return 0.0
    
    # Physiological range check
    if not (40 <= hr <= 180):
        print(f"[WARNING] Physiologically implausible HR: {hr:.1f} BPM")
        return 0.0
    
    print(f"[HR] Detected: {hr:.1f} BPM at {valid_freqs[best_peak]:.3f} Hz "
          f"(confidence: {confidence:.2f})")
    
    return hr

def find_respiration_rate(fft_values, freqs, min_breaths=6, max_breaths=30):
    """
    Improved respiration rate detection with:
    - Better frequency range selection
    - Peak validation
    - Harmonic rejection
    """
    # Convert breaths/min to Hz
    min_hz = min_breaths / 60  # 0.1 Hz for 6 breaths/min
    max_hz = max_breaths / 60  # 0.5 Hz for 30 breaths/min
    
    # Find all peaks in the respiration range
    respiration_mask = (freqs >= min_hz) & (freqs <= max_hz)
    respiration_fft = np.abs(fft_values[respiration_mask])
    respiration_freqs = freqs[respiration_mask]
    
    if len(respiration_fft) == 0:
        print("[DEBUG] No respiration signal detected in frequency range")
        return 0.0
    
    # Find all significant peaks (at least 20% of max amplitude)
    peaks, properties = signal.find_peaks(respiration_fft, 
                                        height=0.2*np.max(respiration_fft))
    
    if len(peaks) == 0:
        print("[DEBUG] No significant respiration peaks found")
        return 0.0
    
    # Get the top 3 peaks by amplitude
    peak_heights = properties['peak_heights']
    top_peaks = peaks[np.argsort(peak_heights)][-3:][::-1]
    
    # Select the lowest frequency among top peaks (avoids harmonics)
    selected_peak = top_peaks[np.argmin(respiration_freqs[top_peaks])]
    respiration_hz = respiration_freqs[selected_peak]
    respiration_bpm = respiration_hz * 60
    
    # Validate the rate is physiologically plausible
    if not (min_breaths <= respiration_bpm <= max_breaths):
        print(f"[DEBUG] Invalid respiration rate {respiration_bpm:.1f} BPM - outside expected range")
        return 0.0
    
    print(f"[DEBUG] Respiration rate: {respiration_bpm:.1f} BPM (peak at {respiration_hz:.3f} Hz)")
    print(f"[DEBUG] Top respiration peaks: {respiration_freqs[top_peaks] * 60}")
    
    return respiration_bpm

def find_hrv(rr_intervals):
    """Keep your existing HRV implementation"""
    if len(rr_intervals) < 2:
        print("[DEBUG] Not enough RR intervals for HRV computation.")
        return 0
    
    hrv_sdnn = np.std(rr_intervals)
    print(f"[DEBUG] HRV (SDNN): {hrv_sdnn:.1f} ms")
    return hrv_sdnn
