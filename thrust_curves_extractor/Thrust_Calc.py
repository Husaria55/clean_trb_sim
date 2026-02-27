import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, savgol_filter
from scipy.stats import t
from scipy.optimize import curve_fit
import glob
import os

"""
Script to calculate expected thrust with uncertainty, 
based on data measured from hot fire by tensometer (in mass)
Just keep data files in 'mass_data' folder
"""

# ============================================================
# USER SETTINGS
# ============================================================
DATA_DIR = "thrust_curves_extractor\\mass_data"  # folder with raw files (no headers)
OUTPUT_CSV = "expected_thrust.csv"

TIME_STEP = 0.002  # common time grid [s]
GRAVITY = 9.80665  # m/s^2

FILTER_CUTOFF = 30  # Hz (None to disable)
SENSOR_STD_FORCE = 5.0  # N (tensometer uncertainty)
SMOOTH_WINDOW = 51  # must be odd, try 31–101
SMOOTH_POLY = 3  # 2–4 recommended

PRE_IGN_AVG_TIME = 5  # seconds before ignition for weight estimate
POST_BURN_AVG_TIME = 5  # seconds after burn for weight estimate

# --- SETTINGS FOR MASS HANDLING ---
# "variable": Compensates for mass loss linearly over time
# "constant": Assumes initial mass remains constant throughout the burn
MASS_LOSS_COMPENSATION = "variable"
# Set to a specific number (e.g., 58.0) to override auto-detection
# Set to None to calculate it automatically from the pre-ignition average
INITIAL_MASS = None

# --- SETTINGS FOR SHIFT & TRIM ---
TIME_SHIFT = 0.0  # [s] Shift the entire final time axis (e.g., 2.0 starts the burn at t=2)
TIME_TRIM_START = None  # [s] Cut off data before this time (None = keep all). Applied AFTER shift.
TIME_TRIM_END = None  # [s] Cut off data after this time (None = keep all). Applied AFTER shift.


# ============================================================

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def lowpass_filter(signal, fs, cutoff):
    [b, a] = butter(4, cutoff / (0.5 * fs), btype='low', output='ba')
    return filtfilt(b, a, signal)


def detect_ignition_and_burn_end(time, force, threshold=20.0, window_size=25):
    """
    Detects ignition and burn end using a moving standard deviation or
    sustained threshold to avoid noise triggers.
    """
    # 1. Smooth slightly to remove high-frequency spikes for detection
    smooth_force = np.convolve(force, np.ones(window_size) / window_size, mode='same')

    # 2. Ignition: Look for when force drops significantly below the baseline
    # We look for a sustained drop (e.g., force stays below threshold for 10 samples)
    ign_idx = None
    for i in range(window_size, len(smooth_force) - 10):
        # Assuming thrust shows as negative force (tensometer compression/tension)
        if np.all(smooth_force[i:i + 10] < -threshold):
            ign_idx = i
            break

    if ign_idx is None:
        raise RuntimeError("Ignition not detected. Check threshold or signal polarity.")

    # 3. Burn End: Look for when the 'slope' flattens out and returns to baseline
    # We look forward from ignition
    end_idx = None
    for i in range(ign_idx + 100, len(smooth_force) - 5):
        # End is when force returns toward zero (weight baseline) and stays stable
        if smooth_force[i] >= -threshold:
            end_idx = i
            break

    if end_idx is None:
        raise RuntimeError("Burn end not detected.")

    return ign_idx, end_idx


def remove_unphysical_drops(
        thrust,
        time,
        drop_threshold=-100,  # Absolute drop in Newtons (adjust to your expected thrust)
        recovery_fraction=0.8,  # Must recover to 80% of pre-drop value
        max_recovery_time=0.5  # Recovery must happen within 0.5 seconds
):
    thrust = thrust.copy()
    n = len(thrust)
    i = 1

    while i < n - 1:
        # Check absolute difference between current and previous point
        delta = thrust[i] - thrust[i - 1]

        # If the drop is larger than our threshold (e.g., -100N)
        if delta < drop_threshold:
            pre_drop_value = thrust[i - 1]
            drop_start_idx = i - 1

            # Look ahead for the recovery point
            found_recovery = False
            for j in range(i + 1, n):
                # Stop looking if we exceed the time limit
                if time[j] - time[drop_start_idx] > max_recovery_time:
                    break

                # Recovery condition: signal returns to near previous levels
                if thrust[j] >= recovery_fraction * pre_drop_value:
                    # Linear interpolation across the "gap"
                    t_segment = [time[drop_start_idx], time[j]]
                    y_segment = [thrust[drop_start_idx], thrust[j]]

                    mask = (time >= time[drop_start_idx]) & (time <= time[j])
                    thrust[mask] = np.interp(time[mask], t_segment, y_segment)

                    i = j  # Fast-forward to the end of the fix
                    found_recovery = True
                    break

            if found_recovery:
                continue

        i += 1
    return thrust


def equation(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d


# ============================================================
# LOAD, PROCESS EACH RUN
# ============================================================
runs = []

files = sorted(glob.glob(os.path.join(DATA_DIR, "*")))

if len(files) < 1:
    raise RuntimeError("At least one hot-fire file required in the directory.")

# Flag to check if we are doing a single run
SINGLE_FILE_MODE = (len(files) == 1)

time_end_ign_longest = 0
time_end_all = []
for file in files:
    data = np.loadtxt(file)
    time = data[:, 0]
    force_meas = data[:, 1]

    if FILTER_CUTOFF is not None:
        fs = 1.0 / np.mean(np.diff(time))
        force_meas = lowpass_filter(force_meas, fs, FILTER_CUTOFF)

    ign_idx, end_idx = detect_ignition_and_burn_end(time, force_meas)

    t_ign = time[ign_idx]
    t_end = time[end_idx]
    time_end_all.append(t_end - t_ign)
    if t_end > time_end_ign_longest:
        time_end_ign_longest = t_end

    burn_time = t_end - t_ign
    print(f"File: {os.path.basename(file)} | Ignition time: {t_ign:.3f} | End time: {t_end:.3f}")

    if INITIAL_MASS is not None:
        W_start = INITIAL_MASS
    else:
        pre_mask = (time >= t_ign - PRE_IGN_AVG_TIME) & (time < t_ign)
        W_start = np.mean(force_meas[pre_mask])

    post_mask = (time > t_end) & (time <= t_end + POST_BURN_AVG_TIME)
    W_end = np.mean(force_meas[post_mask])

    time = time - t_ign

    burn_mask = (time >= 0.0) & (time <= burn_time)
    time = time[burn_mask]
    force_meas = force_meas[burn_mask]

    if MASS_LOSS_COMPENSATION == "variable":
        W_t = W_start + (W_end - W_start) * (time / burn_time)
    elif MASS_LOSS_COMPENSATION == "constant":
        W_t = np.full_like(time, W_start)
    else:
        raise ValueError("Invalid MASS_LOSS_COMPENSATION setting.")

    print(f"  -> Average Comp Weight: {W_t.mean():.2f}")

    thrust = -(force_meas - W_t) * GRAVITY
    runs.append((time, thrust))

# ============================================================
# COMMON TIME GRID
# ============================================================
t_end_common = max(run[0][-1] for run in runs)
time_grid = np.arange(0.0, t_end_common, TIME_STEP)

# ============================================================
# INTERPOLATE
# ============================================================
thrust_matrix = []
for i, (time, thrust) in enumerate(runs):
    time_mask = time <= time_end_all[i]
    interp = interp1d(
        time[time_mask],
        thrust[time_mask],
        kind="linear",
        fill_value=np.nan,
        bounds_error=False
    )
    thrust_interp = interp(time_grid)
    thrust_interp[time_grid > time_end_all[i]] = np.nan
    thrust_matrix.append(thrust_interp)

thrust_matrix = np.array(thrust_matrix)

# ============================================================
# STATISTICS
# ============================================================
mean_thrust = np.nanmean(thrust_matrix, axis=0)
mean_thrust = remove_unphysical_drops(mean_thrust, time_grid)

if SINGLE_FILE_MODE:
    # Skip std deviation and t-factor math if there's only one file
    total_std = np.zeros_like(mean_thrust)
    confidence_band = np.zeros_like(mean_thrust)
    print("Single file mode active: Standard deviation calculations omitted.")
else:
    std_thrust = np.nanstd(thrust_matrix, axis=0, ddof=1)
    total_std = np.sqrt(std_thrust ** 2 + SENSOR_STD_FORCE ** 2)

    N = np.sum(~np.isnan(thrust_matrix), axis=0)
    df = np.maximum(N - 1, 1)
    t_factor = t.ppf(0.975, df=df)

    confidence_band = np.full_like(mean_thrust, 200.0)
    valid = N > 1
    ci = t_factor[valid] * total_std[valid] / np.sqrt(N[valid])
    confidence_band[valid] = np.minimum(ci, 200.0)

mean_thrust = np.maximum(mean_thrust, 0.0)

# Smooth expected thrust curve
valid_mask = ~np.isnan(mean_thrust)
mean_thrust_smooth = mean_thrust.copy()
mean_thrust_smooth[valid_mask] = savgol_filter(
    mean_thrust[valid_mask],
    SMOOTH_WINDOW,
    SMOOTH_POLY
)

mean_thrust = mean_thrust_smooth

# ============================================================
# APPLY TIME SHIFT AND TRIM
# ============================================================
time_grid = time_grid + TIME_SHIFT

trim_start = TIME_TRIM_START if TIME_TRIM_START is not None else time_grid[0]
trim_end = TIME_TRIM_END if TIME_TRIM_END is not None else time_grid[-1]
trim_mask = (time_grid >= trim_start) & (time_grid <= trim_end)

time_grid = time_grid[trim_mask]
mean_thrust = mean_thrust[trim_mask]
total_std = total_std[trim_mask]
confidence_band = confidence_band[trim_mask]
thrust_matrix = thrust_matrix[:, trim_mask]

# ============================================================
# CURVE FITTING
# ============================================================
fit_mask = time_grid < (9 + TIME_SHIFT)
if np.any(fit_mask):
    [popt, pcov] = curve_fit(equation, time_grid[fit_mask], mean_thrust[fit_mask])
else:
    popt = [0, 0, 0, 0]

# ============================================================
# EXPORT TO CSV
# ============================================================
fitted_thrust = equation(time_grid, *popt)

df_out = pd.DataFrame({
    "time_s": time_grid,
    "mean_thrust_N": mean_thrust,
    "fitted_thrust_N": fitted_thrust,
    "std_thrust_N": total_std,
    "ci95_lower_N": mean_thrust - confidence_band,
    "ci95_upper_N": mean_thrust + confidence_band
})
#df_out.to_csv(OUTPUT_CSV, index=False)

thrust_out = pd.DataFrame({"time_s": time_grid, "thrust": mean_thrust})
thrust_out.to_csv("thrust_curves//mean_thrust.csv", index=False, header=False)

# unc_out = pd.DataFrame({"time_s": time_grid, "uncertainty": confidence_band})
# unc_out.to_csv("thrust_curves//mean_thrust_uncertainty.csv", index=False, header=False)

fitted_out = pd.DataFrame({"time_s": time_grid, "fitted_thrust": fitted_thrust})
fitted_out.to_csv("thrust_curves//fitted_thrust.csv", index=False, header=False)

# ============================================================
# PLOT
# ============================================================
plt.figure(figsize=(10, 6))

for run in thrust_matrix:
    plt.plot(time_grid, run, color="gray", alpha=0.3)

plt.plot(time_grid, mean_thrust, "k", linewidth=2, label="Expected Thrust")
plt.plot(time_grid, fitted_thrust, "r", linewidth=2, label="Fitted Thrust")

# Only plot confidence intervals if we have more than one file
if not SINGLE_FILE_MODE:
    plt.fill_between(
        time_grid,
        mean_thrust - confidence_band,
        mean_thrust + confidence_band,
        color="blue",
        alpha=0.3,
        label="95% Confidence Interval"
    )

plt.xlabel("Time [s]")
plt.ylabel("Thrust [N]")
plt.title(f"Expected Thrust (Trimmed/Shifted | Comp: {MASS_LOSS_COMPENSATION.capitalize()})")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print(f"\nOriginal common burn time: {t_end_common:.3f} s")
print(f"Exported data range: {time_grid[0]:.3f} s to {time_grid[-1]:.3f} s")