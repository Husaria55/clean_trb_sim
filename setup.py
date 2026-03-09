"""Module for creating environment, fluids, tanks, motor, rocket, and flight objects for the rocket simulation."""
import numpy as np
from CoolProp.CoolProp import PropsSI
from rocketpy import Rocket, Environment, Flight, LiquidMotor, Fluid, CylindricalTank, MassFlowRateBasedTank
import config as cfg

def create_environment(date=cfg.ENV_DATE_TOMORROW, lat=cfg.ENV_LAT_FAR_OUT, lon=cfg.ENV_LON_FAR_OUT, elevation=cfg.ENV_ELEVATION_FAR_OUT):
    """Creates the Environment object. It uses external API for fetching weather data, currently it fails to work"""
    env = Environment(date=date, latitude=lat, longitude=lon, elevation=elevation)
    env.set_atmospheric_model(type=cfg.ENV_ATM_MODEL_TYPE, file=cfg.ENV_ATM_MODEL_FILE)
    env.max_expected_height = cfg.ENV_MAX_HEIGHT
    return env


def get_env_reanalysis(date, latitude=cfg.ENV_LAT_FAR_OUT, longitude=cfg.ENV_LON_FAR_OUT, elevation=cfg.ENV_ELEVATION_FAR_OUT, max_expected_height=cfg.ENV_MAX_HEIGHT):
    """Creates an Environment object using reanalysis data."""
    env = Environment(
        date=date,
        latitude=latitude,
        longitude=longitude,
        elevation=elevation,
        max_expected_height=max_expected_height,
    )
    env.set_atmospheric_model(
        type="Reanalysis",
        file="data_for_weather_reanalysis.nc",
        dictionary="ECMWF",
    )
    return env


def create_fluids(p_0=cfg.P_0, ethanol_temp=cfg.ETHANOL_TEMPERATURE):
    """Creates Fluid objects and calculates densities using CoolProp.
    Returns tuple: (oxidizer_liquid, oxidizer_gas, fuel_liquid, fuel_gas)
    """
    # Calculate densities
    n2o_liq_rho = PropsSI("D", "P", p_0, "Q", 0, "NitrousOxide")
    n2o_gas_rho = PropsSI("D", "P", p_0, "Q", 1, "NitrousOxide")
    eth_liq_rho = PropsSI("D", "P", p_0, "T", ethanol_temp, "Ethanol")
    eth_gas_rho = PropsSI("D", "P", p_0 - 3e5, "Q", 1, "Ethanol") # -3e5 just to avoid error, we assume nevertheless that ethanol gas mass is 0kg

    oxidizer_liq = Fluid(name="N2O_l", density=n2o_liq_rho)
    oxidizer_gas = Fluid(name="N2O_g", density=n2o_gas_rho)
    fuel_liq = Fluid(name="ethanol_l", density=eth_liq_rho)
    fuel_gas = Fluid(name="ethanol_g", density=eth_gas_rho)

    return oxidizer_liq, oxidizer_gas, fuel_liq, fuel_gas


def create_tanks(
    total_ox_mass=cfg.TOTAL_OXIDIZER_MASS,
    flux_time=cfg.FLUX_TIME,
    piston_pos=cfg.PISTON_POSITION,
    p_0=cfg.P_0,
    oxidizer_mass_flow_rate_out=cfg.OXIDIZER_MASS_FLOW_RATE_OUT,
    fuel_mass_flow_rate_out=cfg.FUEL_MASS_FLOW_RATE_OUT,
    start_time=cfg.START_FLUX_TIME
):
    """
    Creates Oxidizer and Fuel Tank objects.
    Recalculates initial mass splits. 
    Returns tuple: (oxidizer_tank, fuel_tank)
    """
    # 1. Get Fluids
    ox_l, ox_g, fuel_l, fuel_g = create_fluids(p_0=p_0)

    # 2. Geometry
    tank_radius = (cfg.EXTERNAL_TANK_DIAMETER - 2 * cfg.THICKNESS_TANK) / 2
    vol_tank = 0.25 * np.pi * (cfg.EXTERNAL_TANK_DIAMETER - 2 * cfg.THICKNESS_TANK)**2 * cfg.TANK_HEIGHT
    vol_piston = 0.25 * np.pi * (cfg.EXTERNAL_TANK_DIAMETER - 2 * cfg.THICKNESS_TANK)**2 * cfg.THICKNESS_PISTON
    
    vol_ox = piston_pos * vol_tank
    vol_fuel = vol_tank - vol_ox - vol_piston

    # 3. Mass
    gas_init_mass_ox = (vol_ox - (total_ox_mass / ox_l.density)) / (1/ox_g.density - 1/ox_l.density)
    liq_init_mass_ox = total_ox_mass - gas_init_mass_ox
    liq_init_mass_fuel = vol_fuel * fuel_l.density
    gas_init_mass_fuel = cfg.GAS_INITIAL_MASS_FUEL

    # 4. Tank Geometries
    adj_height_ox = piston_pos * cfg.TANK_HEIGHT
    adj_height_fuel = cfg.TANK_HEIGHT - adj_height_ox - cfg.THICKNESS_PISTON

    ox_geom = CylindricalTank(radius=tank_radius, height=adj_height_ox)
    fuel_geom = CylindricalTank(radius=tank_radius, height=adj_height_fuel)

    # 5. Create Tank Objects
    mass_flow_rate_liq = oxidizer_mass_flow_rate_out*(liq_init_mass_ox/total_ox_mass) 
    mass_flow_rate_gas = oxidizer_mass_flow_rate_out*(gas_init_mass_ox/total_ox_mass)
    ox_tank = MassFlowRateBasedTank(
        name="oxidizer tank",
        geometry=ox_geom,
        flux_time=(start_time, flux_time+start_time),
        initial_liquid_mass=liq_init_mass_ox,
        initial_gas_mass=gas_init_mass_ox,
        liquid_mass_flow_rate_in=0,
        liquid_mass_flow_rate_out=mass_flow_rate_liq,
        gas_mass_flow_rate_in=0,
        gas_mass_flow_rate_out=mass_flow_rate_gas,
        liquid=ox_l,
        gas=ox_g,
    )

    fuel_tank = MassFlowRateBasedTank(
        name="fuel tank",
        geometry=fuel_geom,
        flux_time=(start_time, flux_time+start_time),
        initial_liquid_mass=liq_init_mass_fuel - 0.00001,
        initial_gas_mass=gas_init_mass_fuel,
        liquid_mass_flow_rate_in=0,
        liquid_mass_flow_rate_out=fuel_mass_flow_rate_out,
        gas_mass_flow_rate_in=0,
        gas_mass_flow_rate_out=0,
        liquid=fuel_l,
        gas=fuel_g,
    )

    return ox_tank, fuel_tank


def create_motor(
        tanks=None,
        thrust_curve_file=cfg.ENGINE_FILE,
        burn_time=cfg.BURN_TIME,
        start_time=cfg.START_FLUX_TIME
        ):
    """
    Creates the LiquidMotor object.
    
    Args:
        tanks: A tuple (oxidizer_tank, fuel_tank). If None, creates default tanks.
        thrust_curve_file: Path to the thrust curve CSV/eng/or anything else too feed thrust curve file.
        burn_time: Total burn time of the motor.
        start_time: Time at which the motor starts burning. 
    Returns:
        motor: A LiquidMotor object with the specified tanks and thrust curve.
    """
    if tanks is None:
        tanks = create_tanks()
    
    ox_tank, fuel_tank = tanks

    motor = LiquidMotor(
        thrust_source=thrust_curve_file, 
        dry_mass=cfg.MOTOR_DRY_MASS,
        dry_inertia=cfg.MOTOR_DRY_INERTIA,
        nozzle_radius=cfg.NOZZLE_RADIUS,
        center_of_dry_mass_position=cfg.CENTER_OF_DRY_MASS_POS,
        nozzle_position=cfg.NOZZLE_POSITION,
        burn_time=(start_time, start_time + burn_time),
        coordinate_system_orientation=cfg.MOTOR_COORD_SYS,
    )
    
    motor.add_tank(tank=ox_tank, position=cfg.TANK_POSITION_OX)
    motor.add_tank(tank=fuel_tank, position=cfg.TANK_POSITION_FUEL)
    
    return motor


def create_rocket(motor=None, mass=cfg.ROCKET_MASS, inertia=cfg.ROCKET_INERTIA, radius=cfg.ROCKET_RADIUS, center_of_mass_without_motor=cfg.CENTER_OF_MASS_NO_MOTOR, power_on_drag=cfg.DRAG_FILE_ON, power_off_drag=cfg.DRAG_FILE_OFF, coordinate_system_orientation=cfg.ROCKET_COORD_SYS, no_main=False, no_drogue=False, main_at_apogee=False):
    """
    Creates the Rocket object with aerodynamic surfaces and parachutes.
    
    Args:
        motor: LiquidMotor object. If None, creates default motor.
        mass: Dry mass of the rocket without motor.
        inertia: Inertia of the rocket without motor.
        radius: Radius of the rocket body.
        center_of_mass_without_motor: Center of mass position of the rocket without motor.
        power_on_drag: File path for power-on drag data.
        power_off_drag: File path for power-off drag data.
        coordinate_system_orientation: Orientation of the rocket's coordinate system.
        no_main: If True, do not add main parachute.
        no_drogue: If True, do not add drogue parachute.
        main_at_apogee: If True, main parachute is deployed at apogee. Set no_main to True if this is True to avoid adding main parachute twice and no_drogue to True to avoid adding drogue parachute if main is at apogee.
    """
    if motor is None:
        motor = create_motor()

    rocket = Rocket(
        radius=radius,
        mass=mass,
        inertia=inertia,
        power_off_drag=power_off_drag,
        power_on_drag=power_on_drag,
        center_of_mass_without_motor=center_of_mass_without_motor,
        coordinate_system_orientation=coordinate_system_orientation,
    )
    
    rocket.add_motor(motor, position=cfg.MOTOR_POSITION) 

    # Aero Surfaces
    rocket.add_nose(length=cfg.NOSE_LENGTH, kind=cfg.NOSE_KIND, position=cfg.NOSE_POSITION)
    rocket.add_trapezoidal_fins(
        n=cfg.FIN_N,
        root_chord=cfg.FIN_ROOT_CHORD,
        tip_chord=cfg.FIN_TIP_CHORD,
        span=cfg.FIN_SPAN,
        sweep_length=cfg.FIN_SWEEP_LENGTH,
        position=cfg.FIN_POSITION,
        cant_angle=cfg.FIN_CANT_ANGLE
    )
    rocket.add_tail(
        top_radius=cfg.TAIL_TOP_RADIUS, 
        bottom_radius=cfg.TAIL_BOTTOM_RADIUS, 
        length=cfg.TAIL_LENGTH, 
        position=cfg.TAIL_POSITION
    )
    rocket.set_rail_buttons(
        upper_button_position=cfg.BUTTON_UPPER_POS,
        lower_button_position=cfg.BUTTON_LOWER_POS,
        angular_position=cfg.BUTTON_ANGULAR_POS
    )

    # Parachutes (Applying drag factor to Cd_s)
    if not no_main:
        rocket.add_parachute(
            name="main",
            cd_s=cfg.MAIN_CD_S,
            trigger=cfg.MAIN_TRIGGER,
            sampling_rate=cfg.MAIN_SAMPLING_RATE,
            lag=cfg.MAIN_LAG,
            noise=cfg.MAIN_NOISE,
            radius=cfg.MAIN_RADIUS, 
            height=cfg.MAIN_HEIGHT, 
            porosity=cfg.MAIN_POROSITY
    )
    
    if not no_drogue:
        rocket.add_parachute(
            name="drogue",
            cd_s=cfg.DROGUE_CD_S,
            trigger=cfg.DROGUE_TRIGGER,
            sampling_rate=cfg.DROGUE_SAMPLING_RATE,
            lag=cfg.DROGUE_LAG,
            noise=cfg.DROGUE_NOISE,
            radius=cfg.DROGUE_RADIUS, 
            height=cfg.DROGUE_HEIGHT, 
            porosity=cfg.DROGUE_POROSITY
        )
    if main_at_apogee:
        rocket.add_parachute(
            name="main",
            cd_s=cfg.MAIN_CD_S,
            trigger="apogee",
            sampling_rate=cfg.MAIN_SAMPLING_RATE,
            lag=cfg.MAIN_LAG,
            noise=cfg.MAIN_NOISE,
            radius=cfg.MAIN_RADIUS, 
            height=cfg.MAIN_HEIGHT, 
            porosity=cfg.MAIN_POROSITY
        )

    return rocket


def create_flight(rocket=None, env=None, inclination=cfg.INCLINATION_ANGLE, heading=cfg.HEADING_ANGLE, rail_length=cfg.ROD_LENGTH):
    """Creates the Flight object."""
    if rocket is None:
        rocket = create_rocket()
    if env is None:
        env = create_environment()

    return Flight(
        rocket=rocket,
        environment=env,
        rail_length=rail_length,
        inclination=inclination,
        heading=heading
    )