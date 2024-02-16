DEFAULT_TOLERANCE = 0.3
DEFAULT_NAME = "Generic Thermostat"
CONF_AWAY_TEMP = "away_temp"
CONF_ECO_TEMP = "eco_temp"
CONF_BOOST_TEMP = "boost_temp"
CONF_COMFORT_TEMP = "comfort_temp"
CONF_HOME_TEMP = "home_temp"
CONF_SLEEP_TEMP = "sleep_temp"
CONF_ACTIVITY_TEMP = "activity_temp"
CONF_HEATER = "heater"
CONF_SENSOR = "target_sensor"
CONF_MIN_TEMP = "min_temp"
CONF_MAX_TEMP = "max_temp"
CONF_TARGET_TEMP = "target_temp"
CONF_AC_MODE = "ac_mode"
CONF_MIN_DUR = "min_cycle_duration"
CONF_COLD_TOLERANCE = "cold_tolerance"
CONF_HOT_TOLERANCE = "hot_tolerance"
CONF_KEEP_ALIVE = "keep_alive"
CONF_INITIAL_HVAC_MODE = "initial_hvac_mode"
CONF_PRECISION = "precision"
CONF_TEMP_STEP = "target_temp_step"
CONF_PRESETS = {
    p: f"{p}_temp"
    for p in (
        PRESET_AWAY,
        PRESET_ECO,
        PRESET_SLEEP,
        PRESET_ACTIVITY,
        PRESET_HOME,
        PRESET_COMFORT,
        PRESET_BOOST,
    )
}
