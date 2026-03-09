"""Functions for extracting data for far-out report"""
from rocketpy import Flight, LiquidMotor, Rocket
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, hilbert, detrend, savgol_filter


def rail_departure_velocity_in_ft_per_sec(flight: Flight) -> float:
    return flight.out_of_rail_velocity * 3.28084


def average_thrust_during_rail_phase(flight: Flight, motor: LiquidMotor, rocket : Rocket, t_start=1.0) -> float:
    """Calculates the average thrust during the rocket's rail phase.

    This determines the average thrust produced from engine ignition 
    until the rocket clears the launch rail.

    Args:
        flight (Flight): The object containing flight telemetry and state data.
        motor (LiquidMotor): The motor object providing thrust data.
        rocket (Rocket): The rocket object providing mass data.
        t_start (float, optional): Engine ignition time in seconds. Defaults to 1.0.

    Returns:
        float: The average thrust during the rail phase in Newtons.
    """
    t_exit = flight.out_of_rail_time 
    g = 9.80665  
    rail_times = np.linspace(t_start, t_exit, 100)

    # Calculate Average Thrust 
    thrust_values = [motor.thrust(t) for t in rail_times]
    avg_thrust = np.mean(thrust_values)

    # Calculate Average Mass during rail phase
    propellant_mass_values = [motor.propellant_mass(t) for t in rail_times]
    avg_propellant_mass = np.mean(propellant_mass_values)
    avg_total_mass = rocket.mass + avg_propellant_mass + 2.7# rocket.mass is dry mass

    # Calculate TWR
    twr_rail_avg = avg_thrust / (avg_total_mass * g)

    return twr_rail_avg


def max_static_margin(flight: Flight) -> float:
    return np.max(np.array(flight.static_margin)[:,1])


def min_static_margin(flight: Flight) -> float:
    return np.min(np.array(flight.static_margin)[:,1])


def max_acceleration(flight: Flight) -> tuple:
    """Returns the tuple: the maximum acceleration during the flight and the time at which it occurs"""
    return flight.max_acceleration, flight.max_acceleration_time


def max_speed(flight: Flight) -> tuple:
    """Returns the tuple: the maximum speed during the flight and the time at which it occurs."""
    return flight.max_speed, flight.max_speed_time


def max_mach_number(flight: Flight) -> float:
    return flight.max_mach_number


def max_dynamic_pressure(flight: Flight) -> tuple:
    """"Returns the tuple: the maximum dynamic pressure during the flight, the time at which it occurs and the altitude at which it occurs."""
    max_dynamic_pressure = flight.max_dynamic_pressure
    max_dynamic_pressure_time = flight.max_dynamic_pressure_time
    altitude_at_max_pressure = flight.z(max_dynamic_pressure_time)
    return max_dynamic_pressure, max_dynamic_pressure_time, altitude_at_max_pressure


def max_acceleration_in_g(flight: Flight) -> tuple:
    """Returns the tuple: the maximum acceleration during the flight in g's and the time at which it occurs."""
    return flight.max_acceleration / 9.80665, flight.max_acceleration_time


def max_acceleration_power_on_in_g(flight: Flight) -> tuple:
    """
    Calculate the maximum acceleration during the powered flight in g's.
    Returns a tuple containing the maximum acceleration in g's and the time at which it occurs in seconds.
    """
    return flight.max_acceleration_power_on / 9.80665, flight.max_acceleration_power_on_time


def max_velocity_in_ft_per_sec(flight: Flight) -> tuple:
    """"Returns the tuple: the maximum speed during the flight in ft/s and the time at which it occurs."""
    return flight.max_speed * 3.28084, flight.max_speed_time   


def max_q_in_psf_and_altitude_in_ft(flight: Flight) -> tuple:
    """"Returns the tuple: the maximum dynamic pressure during the flight in psf, the time at which it occurs in seconds and the altitude at which it occurs in feet."""
    max_dynamic_pressure = flight.max_dynamic_pressure
    max_dynamic_pressure_time = flight.max_dynamic_pressure_time
    altitude_at_max_pressure = flight.altitude(max_dynamic_pressure_time) * 3.28084
    max_dynamic_pressure_psf = max_dynamic_pressure * 0.0208854
    return max_dynamic_pressure_psf, max_dynamic_pressure_time, altitude_at_max_pressure


def max_altitude_in_ft_and_time(flight: Flight) -> tuple:
    """"Returns the tuple: the maximum altitude during the flight in feet and the time at which it occurs in seconds."""
    return flight.apogee * 3.28084, flight.apogee_time


def distance_from_pad(flight: Flight) -> float:
    x_impact = flight.x_impact
    y_impact = flight.y_impact
    distance_from_pad = np.sqrt(x_impact**2 + y_impact**2)
    return distance_from_pad


def max_yaw_moment(flight: Flight) -> tuple:
    """Returns the tuple: the maximum yaw moment during the flight in N⋅m and the time at which it occurs in seconds."""
    m3_array = np.array(flight.M3)
    max_idx = m3_array[:, 1].argmax()
    M3_max = m3_array[max_idx, 1]
    M3_max_time = m3_array[max_idx, 0]

    return M3_max, M3_max_time


def max_pitch_moment(flight: Flight) -> tuple:
    """Returns the tuple: the maximum pitch moment during the flight in N⋅m and the time at which it occurs in seconds."""
    m2_array = np.array(flight.M2)
    max_idx = m2_array[:, 1].argmax()
    max_moment = m2_array[max_idx, 1]
    max_time = m2_array[max_idx, 0]
    
    return max_moment, max_time


def get_df_for_mach_number(flight: Flight) -> pd.DataFrame:
    """
    Get the Mach number data as a DataFrame.
    Args:
        flight (Flight): The Flight object containing the flight data.

    Returns:
        dataframe: A DataFrame containing the time and Mach number data.
    """
    mach_number = flight.mach_number
    mach_number_array = np.array(mach_number)
    mach_df = pd.DataFrame(mach_number_array, columns=['Time', 'Mach Number'])
    return mach_df


def get_aoa_peaks(flight: Flight) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extracts angle of attack data and identifies oscillation peaks between out of rail and apogee times.
    Returns:
        tuple: (time_data, alpha_data, peak_times, peak_values)
    """
    raw_alpha = np.array(flight.angle_of_attack)
    time_values = raw_alpha[:, 0]
    alpha_values = raw_alpha[:, 1] 

    # Filter data from leaving rail to apogee (with buffer)
    mask = (time_values > flight.out_of_rail_time + 0.1) & (time_values < flight.apogee_time - 0.1)
    t_data = time_values[mask]
    alpha_data = alpha_values[mask]

    # Find peaks
    peaks, _ = find_peaks(alpha_data, distance=5) 
    peak_times = t_data[peaks]
    peak_values = alpha_data[peaks]

    return t_data, alpha_data, peak_times, peak_values


def calculate_damping_ratios(peak_times: np.ndarray, peak_values: np.ndarray) -> tuple[list[float], list[float]]:
    """Calculates damping ratios from AoA peaks using the logarithmic decrement method.
    Returns:
        tuple: (damping_ratios, damping_times)
    """
    damping_ratios = []
    damping_times = []

    for i in range(len(peak_values) - 1):
        a1 = peak_values[i]
        a2 = peak_values[i+1]
        
        # Only calculate if amplitude is decaying and significant
        if a1 > a2 and a1 > 0.5: 
            delta = np.log(a1 / a2)
            zeta = 1 / np.sqrt(1 + (2 * np.pi / delta)**2)
            damping_ratios.append(zeta)
            damping_times.append(peak_times[i])

    return damping_ratios, damping_times


def analyze_and_plot_damping(flight: Flight) -> None:
    """Analyzes the damping ratio for a flight, prints validation metrics, and plots results."""
    # 1. Get data
    t_data, alpha_data, peak_times, peak_values = get_aoa_peaks(flight)
    
    # 2. Do the math
    damping_ratios, damping_times = calculate_damping_ratios(peak_times, peak_values)

    # 3. Print validation metrics
    if not damping_ratios:
        print("Error: No valid damping ratios could be calculated.")
        return

    min_zeta = np.min(damping_ratios)
    max_zeta = np.max(damping_ratios)
    t_min = damping_times[np.argmin(damping_ratios)]
    t_max = damping_times[np.argmax(damping_ratios)]

    print(f"Lowest Damping Ratio:  {min_zeta:.4f} at t={t_min:.2f} s")
    print(f"Highest Damping Ratio: {max_zeta:.4f} at t={t_max:.2f} s")
    
    if min_zeta < 0.05:
        print("Validation: Underdamped (< 0.05)")
    else:
        print("Validation: Minimum damping > 0.05. Passed.")

    # 4. Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(t_data, alpha_data, label='Angle of Attack (deg)')
    plt.plot(peak_times, peak_values, "x", color='red', label='Peaks')
    plt.xlabel("Time (s)")
    plt.ylabel("Alpha (deg)")
    plt.title("Angle of Attack Oscillations")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(damping_times, damping_ratios, marker='o')
    plt.xlabel("Time (s)")
    plt.ylabel("Damping Ratio (Zeta)")
    plt.title("Damping Ratio over Time")
    plt.grid(True)
    plt.show()


def get_flight_signal(flight: Flight, signal_name: str = "partial_angle_of_attack") -> tuple[np.ndarray, np.ndarray]:
    """Extracts a specific flight telemetry signal from rail exit to 10s post-exit.
    Returns:
        tuple: (time_values, signal_values) for the specified signal.
    """
    raw_signal = np.array(getattr(flight, signal_name))
    time_values = raw_signal[:, 0]
    signal_values = raw_signal[:, 1]

    t_exit = flight.out_of_rail_time
    t_end_analysis = flight.max_speed_time

    mask = (time_values > t_exit + 0.1) & (time_values < t_end_analysis)
    
    return time_values[mask], signal_values[mask]


def process_analytic_signal(t_data: np.ndarray, signal_data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Applies Hilbert transform to extract the envelope and instantaneous frequency.
     Returns:
        tuple: (detrended_signal, amplitude_envelope, instantaneous_omega)
    """
    signal_detrended = detrend(signal_data, type='constant') 
    analytic_signal = hilbert(signal_detrended)
    amplitude_envelope = np.abs(analytic_signal)

    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    dt = np.mean(np.diff(t_data))
    instantaneous_omega = np.gradient(instantaneous_phase, dt)

    window_len = min(51, len(t_data) // 5) 
    if window_len % 2 == 0: 
        window_len += 2
        
    amplitude_envelope = savgol_filter(amplitude_envelope, window_len, 3)
    instantaneous_omega = savgol_filter(instantaneous_omega, window_len, 3)

    return signal_detrended, amplitude_envelope, instantaneous_omega


def calculate_sliding_damping(t_data: np.ndarray, amplitude_envelope: np.ndarray, instantaneous_omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculates damping ratio over time using a sliding window linear regression.
     Returns:
        tuple: (damping_ratios, damping_times)
    """
    dt = np.mean(np.diff(t_data))
    samples_per_window = int(2.0 / dt) # 2-second window
    
    damping_ratios, damping_times = [], []

    for i in range(0, len(t_data) - samples_per_window, 1):
        t_chunk = t_data[i : i + samples_per_window]
        amp_chunk = amplitude_envelope[i : i + samples_per_window]
        omega_chunk = instantaneous_omega[i : i + samples_per_window]
        
        if np.min(amp_chunk) < 0.1: 
            continue

        log_amp = np.log(amp_chunk)
        slope, _ = np.polyfit(t_chunk - t_chunk[0], log_amp, 1)
        avg_omega = np.mean(omega_chunk)
        
        if avg_omega > 0:
            zeta = -slope / avg_omega
            if -0.1 < zeta < 1.0:
                damping_ratios.append(zeta)
                damping_times.append(np.mean(t_chunk))

    return np.array(damping_ratios), np.array(damping_times)


def analyze_advanced_damping(flight: Flight, signal_name: str = "partial_angle_of_attack") -> None:
    """Extracts, calculates, and plots advanced damping metrics using Hilbert Transform."""
    # 1 & 2. Get and Process Signal
    t_data, signal_data = get_flight_signal(flight, signal_name)
    signal_detrended, amplitude_envelope, inst_omega = process_analytic_signal(t_data, signal_data)
    
    # 3. Calculate Damping
    damping_ratios, damping_times = calculate_sliding_damping(t_data, amplitude_envelope, inst_omega)

    # 4. Plot and Print
    if len(damping_ratios) == 0:
        print("Could not extract valid damping ratios (check thresholds or data quality).")
        return

    print(f"Mean Damping Ratio: {np.mean(damping_ratios):.4f}")
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    ax1.plot(t_data, signal_data, label=f'Raw {signal_name}', alpha=0.5)
    ax1.plot(t_data, signal_detrended, label='Detrended', color='blue')
    ax1.plot(t_data, amplitude_envelope, label='Hilbert Envelope', color='red', linestyle='--')
    ax1.set_ylabel("Angle / Rate")
    ax1.legend(loc='upper right')
    ax1.set_title("Signal Processing")
    ax1.grid(True)
    
    ax2.plot(t_data, inst_omega / (2*np.pi), color='green')
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_title("Instantaneous Frequency")
    ax2.grid(True)
    
    ax3.plot(damping_times, damping_ratios, 'o-', markersize=4, color='purple')
    ax3.set_ylabel(r"Damping Ratio ($\zeta$)")
    ax3.set_xlabel("Time (s)")
    ax3.set_title("Damping Ratio Evolution")
    ax3.grid(True)



def calculate_aero_centers(rocket: Rocket, flight: Flight) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculates center of pressure (CP), center of gravity (CG), and AoA up to max speed.

    Args:
        rocket (Rocket): The rocket object containing physical parameters.
        flight (Flight): The flight object containing state data.

    Returns:
        tuple: A tuple containing arrays for (time, cp_position, cg_position, angle_of_attack).
    """
    time = flight.time[flight.time <= flight.max_speed_time] 
    
    stability = flight.stability_margin(time)
    aoa = flight.angle_of_attack(time) 
    diameter = 2 * flight.rocket.radius
    
    cg_pos = rocket.center_of_mass(time) 
    cp_pos = cg_pos + (stability * diameter)
    
    return time, cp_pos, cg_pos, aoa


def plot_aerodynamic_stability(rocket: Rocket, flight: Flight) -> None:
    """Plots the relationship between Center of Pressure, Center of Gravity, and Angle of Attack."""
    
    time, cp_pos, cg_pos, aoa = calculate_aero_centers(rocket, flight)
    # Plot 1: CP and CG over Time
    plt.figure(figsize=(10, 6))
    plt.plot(time, cp_pos, label='Center of Pressure', color='red')
    plt.plot(time, cg_pos, label='Center of Gravity', color='blue', linestyle='--')
    plt.xlabel("Time (s)")
    plt.ylabel("Position from Origin (m)")
    plt.title("Center of Pressure and Center of Gravity vs time")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot 2: CP vs AoA
    plt.figure(figsize=(10, 6))
    plt.plot(aoa, cp_pos, label='Center of Pressure', color='green')
    plt.xlabel("Angle of Attack (deg)")
    plt.ylabel("Center of Pressure Position (m)")
    plt.title("Center of Pressure vs Angle of Attack")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_cp_vs_mach_number(rocket: Rocket, flight: Flight) -> None:
    cop = rocket.evaluate_center_of_pressure()
    max_mach = flight.max_mach_number
    mach_lines = np.linspace(0, max_mach, 100)
    cop_values = [cop(mach) for mach in mach_lines]
    plt.figure()
    plt.plot(mach_lines, cop_values)
    plt.xlabel("Mach number")   
    plt.ylabel("Center of pressure [m]")
    plt.title("Center of pressure vs Mach number")
    plt.grid()
    plt.show()
