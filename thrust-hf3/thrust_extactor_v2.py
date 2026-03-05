import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.integrate import trapezoid

# --- Configuration ---
INPUT_FILE = 'thrust-hf3\\tenso_processed.txt'
OUTPUT_FILE = 'thrust_curve_N.csv'
PLOT_FILE = 'thrust_analysis_N.png'

# Filter settings
SAMPLING_RATE = 1000  # Hz
CUTOFF_FREQ = 20      # Hz

# Event detection - Automatic Mode
IGNITION_THRESHOLD_PCT = 0.05  # 5% of max thrust to define burn start/end
MIN_BURN_DURATION_MS = 50      # Signal must stay above threshold for 50ms

# Event detection - Manual Mode Override
USE_MANUAL_TIMES = True        # Set to False to use the automatic detection above
MANUAL_START_TIME = 11.7904400000        # Ignition time in seconds
MANUAL_END_TIME = 11.7904400000 + 5.5       # Burnout time in seconds

# Calibration
CALIBRATION_FACTOR = 9.80665   # Adjust if raw data isn't in kg

def apply_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def main():
    # 1. Load Data
    print(f"Loading data from {INPUT_FILE}...")
    data = np.loadtxt(INPUT_FILE)
    t = data[:, 0]
    raw_measurement = data[:, 1]

    # 2. Taring (Zero-Offset)
    tare_idx = int(SAMPLING_RATE * 1.0) # Average first 1 second
    tare_value = np.mean(raw_measurement[:tare_idx])
    tared_measurement = raw_measurement - tare_value

    # 3. Convert to Newtons
    tared_force_N = tared_measurement * CALIBRATION_FACTOR

    # 4. Filtering
    print("Applying low-pass filter...")
    filtered_force_N = apply_lowpass_filter(tared_force_N, CUTOFF_FREQ, SAMPLING_RATE)

    # 5. Autodetect Force Direction
    if abs(np.min(filtered_force_N)) > abs(np.max(filtered_force_N)):
        print("Note: Inverting signal to make thrust positive.")
        filtered_force_N = -filtered_force_N

    # 6. Event Detection (Manual or Automatic)
    if USE_MANUAL_TIMES:
        print(f"Using manual burn times: {MANUAL_START_TIME}s to {MANUAL_END_TIME}s")
        # Find the indices closest to the specified manual times
        start_idx = np.searchsorted(t, MANUAL_START_TIME)
        end_idx = np.searchsorted(t, MANUAL_END_TIME)
        
        # Ensure indices are within bounds
        start_idx = max(0, min(start_idx, len(t) - 1))
        end_idx = max(0, min(end_idx, len(t) - 1))
        
        if start_idx >= end_idx:
            raise ValueError("MANUAL_START_TIME must be strictly less than MANUAL_END_TIME.")
            
    else:
        print("Using automatic event detection...")
        max_force = np.max(filtered_force_N)
        threshold = IGNITION_THRESHOLD_PCT * max_force
        
        active_indices = np.where(filtered_force_N > threshold)[0]
        if len(active_indices) == 0:
            raise ValueError("No thrust detected above the threshold.")
            
        min_burn_samples = int((MIN_BURN_DURATION_MS / 1000.0) * SAMPLING_RATE)
        
        # Find start
        start_idx = active_indices[0]
        for i in range(len(active_indices) - min_burn_samples):
            if active_indices[i + min_burn_samples] == active_indices[i] + min_burn_samples:
                start_idx = active_indices[i]
                break
                
        # Find end
        end_idx = active_indices[-1]
        for i in range(len(active_indices) - 1, min_burn_samples - 1, -1):
            if active_indices[i - min_burn_samples] == active_indices[i] - min_burn_samples:
                end_idx = active_indices[i]
                break

    # 7. Mass Compensation
    # Average the last 1 second of the recording to find the post-burn weight drop
    post_burn_value_N = np.mean(filtered_force_N[-1000:]) 
    
    baseline_drift_N = np.zeros_like(filtered_force_N)
    baseline_drift_N[:start_idx] = 0
    baseline_drift_N[start_idx:end_idx] = np.linspace(0, post_burn_value_N, end_idx - start_idx)
    baseline_drift_N[end_idx:] = post_burn_value_N
    
    compensated_force_N = filtered_force_N - baseline_drift_N

    # 8. Extract Active Burn Data & Calculate Impulse
    burn_time = t[start_idx:end_idx] - t[start_idx] 
    
    # Force output to be strictly positive
    burn_thrust_N = np.maximum(0, compensated_force_N[start_idx:end_idx])
    
    total_impulse_Ns = trapezoid(burn_thrust_N, burn_time)
    max_thrust_N = np.max(burn_thrust_N)
    avg_thrust_N = total_impulse_Ns / burn_time[-1] if burn_time[-1] > 0 else 0

    print("\n--- Motor Performance Summary ---")
    print(f"Burn Duration: {burn_time[-1]:.3f} s")
    print(f"Max Thrust:    {max_thrust_N:.2f} N")
    print(f"Avg Thrust:    {avg_thrust_N:.2f} N")
    print(f"Total Impulse: {total_impulse_Ns:.2f} N*s")
    print(f"Mass Lost:     {abs(post_burn_value_N / CALIBRATION_FACTOR):.3f} kg") 
    print("---------------------------------\n")

    # 9. Export for Simulation 
    np.savetxt(OUTPUT_FILE, 
               np.column_stack((burn_time, burn_thrust_N)), 
               delimiter=',', 
               header='Time(s),Thrust(N)', 
               comments='')
    print(f"Cleaned thrust curve exported to {OUTPUT_FILE}")

    # 10. Visualization 
    plt.figure(figsize=(10, 6))
    
    plt.plot(burn_time, burn_thrust_N, label='True Thrust (N)', color='red', linewidth=2)
    plt.fill_between(burn_time, burn_thrust_N, color='red', alpha=0.1, label=f'Total Impulse: {total_impulse_Ns:.1f} Ns')
    
    mode_text = "Manual" if USE_MANUAL_TIMES else "Auto"
    plt.title(f'Static Fire Test - Active Burn ({mode_text} Detection)')
    plt.xlabel('Time from Ignition (s)')
    plt.ylabel('Thrust (N)')
    plt.xlim(0, burn_time[-1])
    plt.ylim(0, max_thrust_N * 1.1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_FILE)
    print(f"Analysis plot saved to {PLOT_FILE}")

if __name__ == '__main__':
    main()