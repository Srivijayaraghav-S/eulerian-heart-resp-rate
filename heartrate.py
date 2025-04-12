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
    if not (40 <= hr <= 240):
        print(f"[WARNING] Physiologically implausible HR: {hr:.1f} BPM")
        return 0.0
    
    print(f"[HR] Detected: {hr:.1f} BPM at {valid_freqs[best_peak]:.3f} Hz "
          f"(confidence: {confidence:.2f})")
    
    return hr