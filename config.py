"""
This module defines all the configuration parameters for the rocket simulation, including environment settings, rocket geometry, motor specifications, and propellant properties.
"""
from datetime import datetime, timedelta
import numpy as np

# --- ENVIRONMENT ---
ENV_DATE_TOMORROW = datetime.now() + timedelta(days=1)
ENV_LAT_FAR_OUT = 35.35
ENV_LON_FAR_OUT = -117.81
ENV_ELEVATION_API = "Open-Elevation"
ENV_ELEVATION_FAR_OUT = 621 # use this, rather than Open-Elevation, to avoid API calls
ENV_ATM_MODEL_TYPE = "Forecast"
ENV_ATM_MODEL_FILE = "GFS" # currently this API fails to work, use reanalysis or standard atmosphere instead
ENV_MAX_HEIGHT = 10000

# --- TANK GEOMETRY ---
EXTERNAL_TANK_DIAMETER = 0.2
TANK_HEIGHT = 1.13
THICKNESS_TANK = 0.005
THICKNESS_PISTON = 0.01

# --- PROPELLANT & THERMODYNAMICS ---
P_0 = 65e5  # after the hf 28.03.2026 set to 65 bar instead of 63 bar
PISTON_POSITION = 0.85
TOTAL_OXIDIZER_MASS = 13 # after hf 28.03.2026 set to 13 kg
FLUX_TIME = 5.193 # after hf 28.03.2026 set to 5.15s
ETHANOL_TEMPERATURE = 300
GAS_INITIAL_MASS_FUEL = 0
OXIDIZER_MASS_FLOW_RATE_OUT = 2.503 # for this mass data it's linear aproximation
FUEL_MASS_FLOW_RATE_OUT = 0.6872 # same as ox
START_FLUX_TIME = 0.807 # thrust curve after hf from open rocket data\AGH-SS_Z4000-17kgload.eng

# --- MOTOR ---
ENGINE_FILE = "data/AGH-SS_Z4000-17kgload.eng" # new thrust curve after hf 28.03.2026, this is taken from hf3
MOTOR_DRY_MASS = 2.7
MOTOR_DRY_INERTIA = (0.02143, 0.02143, 0.005535)
NOZZLE_RADIUS = 0.036
CENTER_OF_DRY_MASS_POS = 0.144
NOZZLE_POSITION = 0
BURN_TIME = 5.193 # after hf 28.03.2026 set to 5.15s
MOTOR_COORD_SYS = "nozzle_to_combustion_chamber"
TANK_POSITION_OX = 1.285
TANK_POSITION_FUEL = 2.01
MOTOR_POSITION = 4.49

# --- ROCKET GEOMETRY & MASS ---
ROCKET_RADIUS = 0.1
ROCKET_MASS = 55.549 # the latest mass (09.03.2026)
ROCKET_INERTIA = (72.515, 72.515, 0.426) # after mass changes, this is taken from open rocket
DRAG_FILE_OFF = "./data/powerondrag.csv" # TODO: move to cfd data 
DRAG_FILE_ON = "./data/powerondrag.csv" # This is from open rocket
CENTER_OF_MASS_NO_MOTOR = 2.75
ROCKET_COORD_SYS = "nose_to_tail"

# --- AERODYNAMICS ---
# Nose Cone
NOSE_LENGTH = 0.7
NOSE_KIND = "lvhaack"
NOSE_POSITION = 0

# Fins
FIN_N = 4
FIN_ROOT_CHORD = 0.287
FIN_TIP_CHORD = 0.084
FIN_SPAN = 0.202
FIN_SWEEP_LENGTH = 0.203
FIN_POSITION = 4.21
FIN_CANT_ANGLE = 0

# Tail
TAIL_TOP_RADIUS = 0.1
TAIL_BOTTOM_RADIUS = 0.065
TAIL_LENGTH = 0.287
TAIL_POSITION = 4.21

# Rail Buttons
BUTTON_UPPER_POS = 2.17
BUTTON_LOWER_POS = 3.5
BUTTON_ANGULAR_POS = 0

# --- PARACHUTES ---
MAIN_CD_S = 12.72
MAIN_TRIGGER = 1000
MAIN_SAMPLING_RATE = 105
MAIN_LAG = 6
MAIN_NOISE = (0, 8.3, 0.5)
MAIN_RADIUS = 2.25
MAIN_HEIGHT = 2.25
MAIN_POROSITY = 0.0432

DROGUE_CD_S = 1.218
DROGUE_TRIGGER = "apogee"
DROGUE_LAG = 1
DROGUE_NOISE = (0, 1.0, 0.2)
DROGUE_RADIUS = 0.76
DROGUE_HEIGHT = 0.76
DROGUE_POROSITY = 0.0432
DROGUE_SAMPLING_RATE = 105

# --- Launch Rod ---
ROD_LENGTH = 15.24
INCLINATION_ANGLE = 90 
HEADING_ANGLE = 0