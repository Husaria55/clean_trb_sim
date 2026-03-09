"""
Standalone simulation script equivalent to far_out_clear.ipynb.
Results are appended to simulation_results.txt.
"""

from rocketpy import Fluid, CylindricalTank, MassFlowRateBasedTank
from CoolProp.CoolProp import PropsSI
import numpy as np
from datetime import datetime
import setup
import excel_sheet_functions as ex

# ---------------------------------------------------------------------------
# Simulation parameters – edit here to customise the run
# ---------------------------------------------------------------------------
parameters = {
    "date": (2025, 5, 30, 15),          # (year, month, day, hour) – reanalysis date
    "piston_position": 0.892,           # fraction of the tank height to the bottom of the piston
    "total_oxidizer_mass": 10.5,        # kg
    "flux_time": 5,                     # s – nominal emptying time used for the fuel tank
    "ox_mass_flow_rate": 1.6,           # kg/s – total oxidizer mass flow rate
    "fuel_mass_flow_rate": 0.5,         # kg/s
    "burn_time": 5.727,                 # s
    "thrust_curve_file": "data/AGH-SS_Z4000-13kgload.eng",
    "dry_mass": 55.567,                 # kg – rocket dry mass (without motor and propellant)
    "power_on_drag_file": "data/cd_or_13kg_on.csv",
    "power_off_drag_file": "data/cd_or_13kg_off.csv",
    "inclination": 90,                  # deg – launch rail inclination
    "output_file": "simulation_results.txt",  # results are appended to this file
}

# ---------------------------------------------------------------------------
# Tank geometry constants (match far_out_clear.ipynb)
# ---------------------------------------------------------------------------
EXTERNAL_TANK_DIAMETER = 0.2   # m
TANK_HEIGHT = 1.13             # m
THICKNESS_TANK = 0.005         # m
THICKNESS_PISTON = 0.01        # m
P_0 = 63e5                     # Pa – initial pressure
ETHANOL_TEMPERATURE = 300      # K
GAS_INITIAL_MASS_FUEL = 0      # kg

# Body length for stability % calculation
BODY_LENGTH = 4.49             # m
NOSE_TO_TAIL = 4.99            # m


# ---------------------------------------------------------------------------
# Helper: build tanks from parameters dict
# ---------------------------------------------------------------------------
def build_tanks(params: dict):
    piston_position     = params["piston_position"]
    total_oxidizer_mass = params["total_oxidizer_mass"]
    flux_time           = params["flux_time"]
    ox_mass_flow_rate   = params["ox_mass_flow_rate"]
    fuel_mass_flow_rate = params["fuel_mass_flow_rate"]

    # Fluid densities
    N2O_liq_density    = PropsSI("D", "P", P_0,       "Q", 0, "NitrousOxide")
    N2O_gas_density    = PropsSI("D", "P", P_0,       "Q", 1, "NitrousOxide")
    ethanol_liq_density = PropsSI("D", "P", P_0 - 1e5, "T", ETHANOL_TEMPERATURE, "Ethanol")
    ethanol_gas_density = PropsSI("D", "P", P_0 - 1e5, "Q", 1, "Ethanol")

    oxidizer_liq = Fluid(name="N2O_l",     density=N2O_liq_density)
    oxidizer_gas = Fluid(name="N2O_g",     density=N2O_gas_density)
    fuel_liq     = Fluid(name="ethanol_l", density=ethanol_liq_density)
    fuel_gas     = Fluid(name="ethanol_g", density=ethanol_gas_density)

    # Volumes
    tank_radius   = (EXTERNAL_TANK_DIAMETER - 2 * THICKNESS_TANK) / 2
    volume_tank   = 0.25 * np.pi * (EXTERNAL_TANK_DIAMETER - 2 * THICKNESS_TANK) ** 2 * TANK_HEIGHT
    volume_oxidizer = piston_position * volume_tank
    volume_fuel     = volume_tank - volume_oxidizer - \
                      0.25 * np.pi * (EXTERNAL_TANK_DIAMETER - 2 * THICKNESS_TANK) ** 2 * THICKNESS_PISTON

    # Initial masses
    gas_initial_mass_ox   = (volume_oxidizer - total_oxidizer_mass / N2O_liq_density) / \
                             (1 / N2O_gas_density - 1 / N2O_liq_density)
    liquid_initial_mass_ox  = total_oxidizer_mass - gas_initial_mass_ox
    liquid_initial_mass_fuel = volume_fuel * ethanol_liq_density

    # Per-phase flow rates (split by mass fraction for oxidizer)
    mass_flow_rate_liq = ox_mass_flow_rate * (liquid_initial_mass_ox / total_oxidizer_mass)
    mass_flow_rate_gas = ox_mass_flow_rate * (gas_initial_mass_ox   / total_oxidizer_mass)

    # Tank heights
    adjusted_height_ox   = piston_position * TANK_HEIGHT
    adjusted_height_fuel = TANK_HEIGHT - adjusted_height_ox - THICKNESS_PISTON

    # Geometries
    oxidizer_tank_geometry = CylindricalTank(radius=tank_radius, height=adjusted_height_ox + 0.00001)
    fuel_tank_geometry     = CylindricalTank(radius=tank_radius, height=adjusted_height_fuel)

    # Tanks
    oxidizer_tank = MassFlowRateBasedTank(
        name="oxidizer tank",
        geometry=oxidizer_tank_geometry,
        flux_time=flux_time + 1.56,        # extra time to fully empty the oxidizer tank
        initial_liquid_mass=liquid_initial_mass_ox,
        initial_gas_mass=gas_initial_mass_ox,
        liquid_mass_flow_rate_in=0,
        liquid_mass_flow_rate_out=mass_flow_rate_liq,
        gas_mass_flow_rate_in=0,
        gas_mass_flow_rate_out=mass_flow_rate_gas,
        liquid=oxidizer_liq,
        gas=oxidizer_gas,
    )

    fuel_tank = MassFlowRateBasedTank(
        name="fuel tank",
        geometry=fuel_tank_geometry,
        flux_time=flux_time,
        initial_liquid_mass=liquid_initial_mass_fuel - 0.00001,
        initial_gas_mass=GAS_INITIAL_MASS_FUEL,
        liquid_mass_flow_rate_in=0,
        liquid_mass_flow_rate_out=fuel_mass_flow_rate,
        gas_mass_flow_rate_in=0,
        gas_mass_flow_rate_out=0,
        liquid=fuel_liq,
        gas=fuel_gas,
    )

    return oxidizer_tank, fuel_tank


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------
def run_simulation(params: dict) -> dict:
    print("=== Setting up environment ===")
    env = setup.get_env_reanalysis(params["date"])

    print("=== Building tanks ===")
    oxidizer_tank, fuel_tank = build_tanks(params)

    print("=== Creating motor ===")
    motor = setup.create_motor(
        tanks=(oxidizer_tank, fuel_tank),
        thrust_curve_file=params["thrust_curve_file"],
        burn_time=params["burn_time"],
    )

    print("=== Creating rocket ===")
    rocket = setup.create_rocket(
        motor=motor,
        mass=params["dry_mass"],
        power_on_drag=params["power_on_drag_file"],
        power_off_drag=params["power_off_drag_file"],
    )

    print("=== Running nominal flight ===")
    flight = setup.create_flight(rocket=rocket, env=env, inclination=params["inclination"])

    # --- Rail / liftoff metrics ---
    twr                       = ex.average_thrust_during_rail_phase(flight, motor, rocket, t_start=0)
    rail_departure_vel_ft     = ex.rail_departure_velocity_in_ft_per_sec(flight)
    rail_departure_vel_m      = flight.out_of_rail_velocity
    max_acc_power_on          = flight.max_acceleration_power_on
    max_acc_power_on_time     = flight.max_acceleration_power_on_time

    # --- Speed / pressure metrics ---
    max_q_time    = flight.max_dynamic_pressure_time
    max_mach_time = flight.max_mach_number_time
    max_q_alt     = flight.altitude(max_q_time)
    max_speed_val = flight.max_speed
    max_speed_time = flight.max_speed_time
    max_mach      = flight.max_mach_number

    # --- Apogee ---
    apogee     = flight.altitude(flight.apogee_time)
    apogee_time = flight.apogee_time

    # --- Stability ---
    cg_from_tail     = NOSE_TO_TAIL - rocket.center_of_mass(0)
    cp_from_tail     = NOSE_TO_TAIL - rocket.cp_position(0)
    static_margin_0  = rocket.stability_margin(0, 0)
    max_stab_pct     = flight.max_stability_margin * 0.2 / BODY_LENGTH * 100
    min_stab_pct     = flight.min_stability_margin * 0.2 / BODY_LENGTH * 100

    # --- Landing distances ---
    dist_nominal = ex.distance_from_pad(flight)

    print("=== Running ballistic flight (no parachutes) ===")
    rocket_ballistic  = setup.create_rocket(motor=motor, no_main=True, no_drogue=True,
                                             mass=params["dry_mass"],
                                             power_on_drag=params["power_on_drag_file"],
                                             power_off_drag=params["power_off_drag_file"])
    flight_ballistic  = setup.create_flight(rocket=rocket_ballistic, env=env, inclination=params["inclination"])
    dist_ballistic    = ex.distance_from_pad(flight_ballistic)

    print("=== Running drogue-only flight ===")
    rocket_drogue     = setup.create_rocket(motor=motor, no_main=True, no_drogue=False,
                                             mass=params["dry_mass"],
                                             power_on_drag=params["power_on_drag_file"],
                                             power_off_drag=params["power_off_drag_file"])
    flight_drogue     = setup.create_flight(rocket=rocket_drogue, env=env, inclination=params["inclination"])
    dist_drogue       = ex.distance_from_pad(flight_drogue)

    print("=== Running main-at-apogee flight ===")
    rocket_main_apogee = setup.create_rocket(motor=motor, no_main=True, no_drogue=True, main_at_apogee=True,
                                              mass=params["dry_mass"],
                                              power_on_drag=params["power_on_drag_file"],
                                              power_off_drag=params["power_off_drag_file"])
    flight_main_apogee = setup.create_flight(rocket=rocket_main_apogee, env=env, inclination=params["inclination"])
    dist_main_apogee   = ex.distance_from_pad(flight_main_apogee)

    # --- Moments ---
    max_yaw_val, max_yaw_time     = ex.max_yaw_moment(flight)
    max_pitch_val, max_pitch_time = ex.max_pitch_moment(flight)

    return {
        # rail
        "twr_at_liftoff": twr,
        "rail_departure_velocity_ft_s": rail_departure_vel_ft,
        "rail_departure_velocity_m_s": rail_departure_vel_m,
        "max_acceleration_power_on_m_s2": max_acc_power_on,
        "max_acceleration_power_on_time_s": max_acc_power_on_time,
        # speed / pressure
        "max_q_time_s": max_q_time,
        "max_mach_time_s": max_mach_time,
        "altitude_at_max_q_m": max_q_alt,
        "max_speed_m_s": max_speed_val,
        "max_speed_time_s": max_speed_time,
        "max_mach_number": max_mach,
        # apogee
        "apogee_m": apogee,
        "apogee_time_s": apogee_time,
        # stability
        "cg_from_tail_m": cg_from_tail,
        "cp_from_tail_m": cp_from_tail,
        "static_stability_margin_cal": static_margin_0,
        "max_static_margin_pct_body": max_stab_pct,
        "min_static_margin_pct_body": min_stab_pct,
        # distances
        "distance_nominal_m": dist_nominal,
        "distance_ballistic_m": dist_ballistic,
        "distance_drogue_only_m": dist_drogue,
        "distance_main_at_apogee_m": dist_main_apogee,
        # moments
        "max_yaw_moment_Nm": max_yaw_val,
        "max_yaw_moment_time_s": max_yaw_time,
        "max_pitch_moment_Nm": max_pitch_val,
        "max_pitch_moment_time_s": max_pitch_time,
    }


def save_results(results: dict, params: dict, output_file: str) -> None:
    """Append simulation results to a text file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep = "=" * 70

    lines = [
        "",
        sep,
        f"Simulation run: {timestamp}",
        sep,
        "--- Parameters ---",
        f"  Date (reanalysis):          {params['date']}",
        f"  Piston position:            {params['piston_position']}",
        f"  Total oxidizer mass:        {params['total_oxidizer_mass']} kg",
        f"  Flux time:                  {params['flux_time']} s",
        f"  Oxidizer mass flow rate:    {params['ox_mass_flow_rate']} kg/s",
        f"  Fuel mass flow rate:        {params['fuel_mass_flow_rate']} kg/s",
        f"  Burn time:                  {params['burn_time']} s",
        f"  Thrust curve:               {params['thrust_curve_file']}",
        f"  Dry mass:                   {params['dry_mass']} kg",
        f"  Inclination:                {params['inclination']} deg",
        "",
        "--- Rail / Liftoff ---",
        f"  TWR at liftoff:                       {results['twr_at_liftoff']:.8f}",
        f"  Rail departure velocity:              {results['rail_departure_velocity_ft_s']:.8f} ft/s",
        f"  Rail departure velocity:              {results['rail_departure_velocity_m_s']:.8f} m/s",
        f"  Max acceleration (power on):          {results['max_acceleration_power_on_m_s2']:.8f} m/s²",
        f"  Time of max acceleration (power on):  {results['max_acceleration_power_on_time_s']:.8f} s",
        "",
        "--- Speed / Dynamic Pressure ---",
        f"  Time of max dynamic pressure:  {results['max_q_time_s']:.4f} s",
        f"  Time of max Mach number:       {results['max_mach_time_s']:.4f} s",
        f"  Altitude at max dynamic pres.: {results['altitude_at_max_q_m']:.4f} m",
        f"  Max speed:                     {results['max_speed_m_s']:.4f} m/s",
        f"  Time of max speed:             {results['max_speed_time_s']:.4f} s",
        f"  Max Mach number:               {results['max_mach_number']:.4f}",
        "",
        "--- Apogee ---",
        f"  Apogee:        {results['apogee_m']:.4f} m",
        f"  Time at apogee:{results['apogee_time_s']:.4f} s",
        "",
        "--- Aerodynamic Stability ---",
        f"  Max static margin (% body length): {results['max_static_margin_pct_body']:.4f} %",
        f"  Min static margin (% body length): {results['min_static_margin_pct_body']:.4f} %",
        f"  CG from tail:                      {results['cg_from_tail_m']:.4f} m",
        f"  CP from tail:                      {results['cp_from_tail_m']:.4f} m",
        f"  Static stability margin:           {results['static_stability_margin_cal']:.4f} cal",
        "",
        "--- Landing Distances ---",
        f"  Nominal (main + drogue):     {results['distance_nominal_m']:.2f} m",
        f"  Ballistic (no parachutes):   {results['distance_ballistic_m']:.2f} m",
        f"  Drogue only:                 {results['distance_drogue_only_m']:.2f} m",
        f"  Main at apogee:              {results['distance_main_at_apogee_m']:.2f} m",
        "",
        "--- Moments ---",
        f"  Max yaw moment:   {results['max_yaw_moment_Nm']:.4f} N·m  at t={results['max_yaw_moment_time_s']:.4f} s",
        f"  Max pitch moment: {results['max_pitch_moment_Nm']:.4f} N·m  at t={results['max_pitch_moment_time_s']:.4f} s",
        sep,
    ]

    with open(output_file, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nResults appended to '{output_file}'")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = run_simulation(parameters)

    # Print summary to console
    print("\n" + "=" * 70)
    print("SIMULATION RESULTS")
    print("=" * 70)
    for key, value in results.items():
        print(f"  {key}: {value}")

    save_results(results, parameters, parameters["output_file"])
