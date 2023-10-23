import streamlit as st
import pydeck as pdk
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import requests
import pandas as pd
import math

from streamlit_folium import st_folium
import folium

from PIL import Image
from io import BytesIO
import base64

import numpy as np

# import plot_climate_and_growth_potential function
def fetch_climate_data_regrow(lat, lon, start_date, end_date, monthly=True, plot=False):
    '''
    Fetching climate data from NasaPower through Regrow's climate service endpoint
    NasaPower variable cheatsheet: https://gist.github.com/abelcallejo/d68e70f43ffa1c8c9f6b5e93010704b8
    T2M	Temperature at 2 Meters	The average air (dry bulb) temperature at 2 meters above the surface of the earth.
    '''
    
     # Set time period
    if isinstance(start_date,datetime):
        start_date = start_date.strftime('%Y-%m-%d')
    if isinstance(end_date,datetime): 
        end_date = end_date.strftime('%Y-%m-%d')
        
    # Construct the API URL
    url = f"http://api.dev.internal:9090/global-climate-service/nasa-power-weather-data"
    params = {
        'lat': lat,
        'lon': lon,
        'start_date': start_date,
        'end_date': end_date,
        'return_npy_file': 'false'
    }

    # Make the API request
    response = requests.get(url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON data
        json_data = response.json()

        # Extract column names and data
        column_names = json_data[0]
        data = json_data[1:]

        # Create a Pandas DataFrame
        df = pd.DataFrame(data, columns=column_names)
        # Rename the columns
        df.rename(columns={
            'T2M': 'tavg',
            'T2M_MAX': 'tmax',
            'T2M_MIN': 'tmin'
        }, inplace=True)

        # Convert 'date' column to datetime type
        df['date'] = pd.to_datetime(df['date'])
        # Set 'date' column as the index
        df.set_index('date', inplace=True)
        
        # if monthly:
        # Resample to monthly frequency, taking the mean of each month
        df_month = df.resample('M').mean()
        df_month['year_month'] = df_month.index.to_period('M')
            
        if plot:
            # Plot line chart including average, minimum and maximum temperature
            df.plot(y=['tavg', 'tmin', 'tmax'])
            plt.show()
        return df, df_month
    else:
        print(f"Failed to retrieve data. HTTP Status Code: {response.status_code}")
        return None, None


## To deploy need to use public endpoint from Nasa power
def fetch_climate_data(lat, lon, start_date, end_date, monthly=True, plot=False):
    '''
    Fetching climate data from NasaPower
    NasaPower variable cheatsheet: https://gist.github.com/abelcallejo/d68e70f43ffa1c8c9f6b5e93010704b8
    T2M	Temperature at 2 Meters	The average air (dry bulb) temperature at 2 meters above the surface of the earth.
    '''
    
     # Set time period
    if isinstance(start_date,datetime):
        start_date = start_date.strftime('%Y-%m-%d')
    if isinstance(end_date,datetime): 
        end_date = end_date.strftime('%Y-%m-%d')
        
    # Construct the API URL
    url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M,T2MDEW,T2MWET,TS,T2M_RANGE,T2M_MAX,T2M_MIN&community=RE"

    params = {
        'latitude': lat,
        'longitude': lon,
        'start': start_date.replace("-",""),
        'end': end_date.replace("-",""),
        'format': 'JSON',
    }

    # Make the API request
    response = requests.get(url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON data
        json_data = response.json()
        
        # Extract relevant data
        t2m_data = json_data['properties']['parameter']['T2M']
        t2m_max_data = json_data['properties']['parameter']['T2M_MAX']
        t2m_min_data = json_data['properties']['parameter']['T2M_MIN']

        # Create DataFrame
        df = pd.DataFrame({
            'date': list(t2m_data.keys()),
            'tavg': list(t2m_data.values()),
            'tmax': list(t2m_max_data.values()),
            'tmin': list(t2m_min_data.values())
        })

        # Convert 'date' column to datetime type and set it as the index
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # if monthly:
        # Resample to monthly frequency, taking the mean of each month
        df_month = df.resample('M').mean()
        df_month['year_month'] = df_month.index.to_period('M')
            
        if plot:
            # Plot line chart including average, minimum and maximum temperature
            df.plot(y=['tavg', 'tmin', 'tmax'])
            plt.show()
        return df, df_month
    else:
        print(f"Failed to retrieve data. HTTP Status Code: {response.status_code}")
        return None, None

# import math
def calculate_growth_potential(temp, optimum_growth_temp, temp_variance, species):
    """
    Calculate the Growth Potential (GP) using the given formula.
    temp: Average monthly temperature in Celsius
    optimum_growth_temp: Optimum Growth Temperature in Celsius, use 20 for c3 and 31 for c4
    temp_variance: Temperature variance. use 10 for c3 and 12 for c4
    species: 'c3' or 'c4'
    """
    # Converting temperatures to Fahrenheit
    temp_f = temp * 1.8 + 32
    optimum_growth_temp_f = optimum_growth_temp * 1.8 + 32
    
    # Calculating the GP using the given formula
    exponent = -0.5 * ((temp_f - optimum_growth_temp_f) / temp_variance) ** 2
    gp = math.exp(exponent)
    
    # Making sure GP is between 0 and 1
    species = species.lower()  # Making it case-insensitive
    if species == 'c3':
        if 100 * gp < 1:
            gp = 0.0
    elif species == 'c4':
        if temp > optimum_growth_temp:
            gp = 1.0
        elif 100 * gp < 1:
            gp = 0.0
    else:
        raise ValueError("Invalid species. Please enter 'c3' or 'c4'.")
    
    return gp

### C and N accumulation
def calculate_uptake_daily(gp, max_n, c_n_ratio, kg_ha=False):
    '''distribute the monthly growth to daily 
    gp: growth potential based on temperature
    max_n: maximum monthly N uptake gN/m2/month, 3.5 g/m2 for C3 and 4 g/m2 for C4
    c_n_ratio: c/n ratio of leaf, default to 20
    '''
    n_uptake_daily = gp * max_n/30 # assuming a 30 days in a month
    c_accumulation_daily = c_n_ratio * n_uptake_daily
    if kg_ha:
        n_uptake_daily = n_uptake_daily * 10
        c_accumulation_daily = c_accumulation_daily * 10
    return n_uptake_daily, c_accumulation_daily

### Dormancy and Regrowth logic
#### Logic
# Update the function to handle multiple years
def dormancy_regrowth_logic(
    min_air_temperature, 
    avg_air_temperature, 
    hemisphere='Northern', 
    is_perennial=True,
    species='c3',
    max_n_month=3.5,
    c_n_ratio=20,
    optimum_growth_temp=20,
    temp_variance=10,
    T_critical=-5.0,
    T_base=0.0,
    TDD_max=2000.0,
    tdd_min=0.0,
    gp_threshold=0.1,
    regrowth_reset_frac=0.1,
    TDD_max_reset_frac=0,
    min_chilling_hours=100, 
    chilling_temp_range=(0, 7),
    consecutive_freezing_days_threshold=7,
    consecutive_warming_days_threshold=7,
    max_consecutive_day_tolerant=3,  # New variable for consecutive day tolerance
    debug=False
):
    '''
    Simulates the perennial growth logic based on temperature-dependent degree days (TDD),
    chilling hours, dormancy, and regrowth conditions.
    
    Notes:
    Simulates the dormancy and regrowth logic of perennial plants based on daily minimum and average air temperature.
    
    1. Winter Dormancy: The plant goes into dormancy if the daily minimum air temperature is less than or equal to a critical 
    temperature T_critical for a certain number of consecutive days (calibrated parameter).
    
    2. Spring Regrowth: The plant comes out of dormancy and regrows in spring if the daily average air temperature is above 
    the base temperature T_base for a consecutive number of days (calibrated parameter). Additionally, a minimum number of 
    chilling hours (calibrated parameter) must be accumulated during the dormancy period for regrowth to occur.
    
    3. Chilling Hours: During the dormant state, the plant accumulates chilling hours based on the daily minimum and average 
    air temperatures. A full day or a half-day of chilling hours is accumulated based on whether the daily minimum or the 
    daily average temperature falls within the chilling temperature range (calibrated parameter).
    
    4. TDD (Thermal Time) Accumulation: The function also calculates the Thermal Time (TDD) accumulation, which is paused 
    during the dormant state and resumes during the growing state. At the time of spring regrowth, the TDD accumulation is 
    set to a percentage of the maximum TDD TDD_max (calibrated parameter).
    
    5. Calibration: The model is designed to be flexible, allowing several parameters to be calibrated, including T_critical,
    T_base, TDD_max, minimum chilling hours, chilling temperature range, and the number of consecutive days for triggering 
    dormancy and regrowth.
    
    6. Debugging: The function includes a debug flag that can be enabled to print detailed information for model diagnostics.
    
    7. Consecutive Days: The variables consecutive_freezing_days and consecutive_warming_days are meant to be truly consecutive.
    A new parameter max_consecutive_day_tolerant specifies the maximum number of days that can pass without meeting the 
    freezing or warming conditions before the consecutive count is reset to zero.
    
    8. Future implementation should include logic for summer dormancy due to high temperatures and low moisture as well as
    hemisphere Handling. Currently winter/spring months are calculated for the Northern and Southern hemisphere but they are
    not used in the logic. The calculation of current month was also simplified.
    
    Parameters:
        - species='c3', 
        max_n_month=3.5,
            c_n_ratio=20,
        - gp_threshold
        - optimum_growth_temp and temp_variance are used for GP calcs
        - Various temperature thresholds and constants (T_critical, T_base, etc.) (in Celsius) for dormancy logic
        - Hemisphere to differentiate winter and spring months
        - is_perennial flag to specify perennial plants
        - Chilling hour requirements (min_chilling_hours, chilling_temp_range)
        - Consecutive day thresholds for dormancy and regrowth
        - tdd_min: Minimum TDD accumulation before dormancy and regrowth logic kicks in (for calibration)
        - regrowth_reset_frac: variable for setting TDD accumulation at the time of regrowth
        - TDD_max_reset_frac: variable for resetting TDD accumulation when it reaches TDD_max
        - max_consecutive_day_tolerant: Maximum number of days that can pass without meeting the consecutive freezing or warming
    condition before the count is reset to zero.
        - debug: If True, prints debug information during simulation

    Returns:
        An array of TDD accumulations for each day.
    '''
    
    # Initializations
    days_in_simulation = len(min_air_temperature)
    days_in_year = 365
    consecutive_freezing_days = 0
    consecutive_warming_days = 0
    days_since_last_freezing = 0
    days_since_last_warming = 0
    tdd_accumulation = 0
    chilling_hours = 0
    gp = 0
    c_accumulation = 0
    c_accumulation_daily=0
    
    # Initialize output array for TDD accumulation
    tdd_accumulation_array = np.zeros(days_in_simulation)
    
    # Initialize arrays to store additional variables
    is_dormant_array = np.zeros(days_in_simulation, dtype=bool)
    chilling_hours_array = np.zeros(days_in_simulation)
    consecutive_freezing_days_array = np.zeros(days_in_simulation)
    consecutive_warming_days_array = np.zeros(days_in_simulation)
    gp_array = np.zeros(days_in_simulation)
    c_accumulation_array = np.zeros(days_in_simulation)
    
    # Initialize dormancy flag
    is_dormant = False
    
    if is_perennial:  # Only execute this logic for perennial plants
    
        # Loop through each day in multiple years
        for day in range(days_in_simulation):
            # Calculate the current year (0-based) and day within the year
            current_year = day // days_in_year
            day_within_year = day % days_in_year
            
            # Current temperature and month
            temperature = min_air_temperature[day]
            avg_temperature = avg_air_temperature[day]
            current_month = (day_within_year // 30) % 12 + 1  # Simplified; each month has 30 days
            
            # Calculate growth potential
            gp = calculate_growth_potential(avg_temperature, optimum_growth_temp, temp_variance, species)
            n_uptake_daily, c_accumulation_daily = calculate_uptake_daily(gp, max_n_month, c_n_ratio)
            
            # Debug prints
            if debug:
                print(f"Year: {current_year+1}, Day: {day_within_year+1}, Month: {current_month}, Min Air Temp: {round(temperature, 1)}, Avg Air Temp: {round(avg_temperature, 1)}, Is Dormant: {is_dormant}, Chilling Hours: {chilling_hours}")

            # Check for winter dormancy condition
            if not is_dormant:
                if tdd_accumulation >= tdd_min: # if the plant hasn't grown yet, don't check for dormancy
                    if temperature <= T_critical:
                        consecutive_freezing_days += 1
                        days_since_last_freezing = 0  # Reset days since last freezing
                        if debug:
                            print(f"  - Checking for winter dormancy: {consecutive_freezing_days} freezing days.")
                    else:
                        days_since_last_freezing += 1  # Increment days since last freezing

                    # Reset consecutive_freezing_days if tolerance is exceeded
                    if days_since_last_freezing > max_consecutive_day_tolerant:
                        consecutive_freezing_days = 0

                    # Check if winter dormancy is triggered
                    if consecutive_freezing_days >= consecutive_freezing_days_threshold:
                        is_dormant = True
                        consecutive_freezing_days = 0
                        if debug:
                            print("  - Winter dormancy triggered.")
                            
            else:  # if is_dormant
                
                # Check for chilling hours accumulation
                if chilling_temp_range[0] <= temperature <= chilling_temp_range[1]:
                    chilling_hours += 24  # Full day contributes to chilling hours
                    if debug:
                        print("  - Full day contributes to chilling hours.")
                elif chilling_temp_range[0] <= avg_temperature <= chilling_temp_range[1]:
                    chilling_hours += 12  # Half day contributes to chilling hours
                    if debug:
                        print("  - Half day contributes to chilling hours.")
                
                # Check for spring regrowth condition
                if avg_temperature >= T_base:
                    consecutive_warming_days += 1
                    days_since_last_warming = 0  # Reset days since last warming
                    if debug:
                        print(f"  - Checking for spring regrowth: {consecutive_warming_days} warming days.")
                else:
                    days_since_last_warming += 1  # Increment days since last warming

                # Reset consecutive_warming_days if tolerance is exceeded
                if days_since_last_warming > max_consecutive_day_tolerant:
                    consecutive_warming_days = 0
                
                # Check if spring regrowth is triggered (only if plant is dormant and chilling hour requirements are met)
                if consecutive_warming_days >= consecutive_warming_days_threshold and chilling_hours >= min_chilling_hours:
                    tdd_accumulation = regrowth_reset_frac * TDD_max  # Use regrowth_reset_frac
                    is_dormant = False
                    consecutive_warming_days = 0
                    chilling_hours = 0  # Reset chilling hours
                    if debug:
                        print("  - Spring regrowth triggered.")
                
            # Update TDD accumulation based on dormancy status
            if is_dormant or gp < gp_threshold:
                tdd_accumulation += 0
                c_accumulation +=0
                if is_dormant and c_accumulation > 10: # 10gC/m2 100kg/ha 
                    c_accumulation -= max_n_month * c_n_ratio * TDD_max_reset_frac
            else:
                tdd_accumulation += max(0, avg_temperature - temperature ) # should it be min_temp or T_base??
                c_accumulation += c_accumulation_daily
            
            # Check if TDD accumulation exceeds TDD_max
            if tdd_accumulation >= TDD_max:
                tdd_accumulation = TDD_max_reset_frac * TDD_max  # Reset to TDD_max_reset_frac
                # c_accumulation = c_accumulation * TDD_max_reset_frac
                
            if debug:
                print(f"  - TDD accumulation updated: {tdd_accumulation}")
            
            # Update TDD accumulation array
            tdd_accumulation_array[day] = tdd_accumulation
            
            # Update other arrays for visualisation
            is_dormant_array[day] = is_dormant
            chilling_hours_array[day] = chilling_hours
            consecutive_freezing_days_array[day] = consecutive_freezing_days
            consecutive_warming_days_array[day] = consecutive_warming_days
            gp_array[day] = gp
            c_accumulation_array[day] = c_accumulation
    
    return tdd_accumulation_array, is_dormant_array, chilling_hours_array, consecutive_freezing_days_array, consecutive_warming_days_array, gp_array, c_accumulation_array

#### Plotting logic
# Visualization function to plot important variables
# Visualization function to plot important variables
def visualize_dormancy_regrowth_logic(min_air_temperature,
                                      avg_air_temperature,
                                      tdd_accumulation,
                                      is_dormant,
                                      chilling_hours,
                                      consecutive_freezing_days,
                                      consecutive_warming_days,
                                      gp, c_accumulation,
                                     show=False,
                                     title=''):
    # Create a figure with multiple subplots
    fig, axs = plt.subplots(8, 1, figsize=(15, 20))
    
    
    # Plot Min and Avg Air Temperature
    i=0
    axs[i].plot(min_air_temperature, label='Min Air Temperature (C)', color='blue')
    axs[i].plot(avg_air_temperature, label='Avg Air Temperature (C)', color='orange')
    axs[i].axhline(y=-5, color='r', linestyle='--', label='T_critical (-5 C)')
    axs[i].set_title('Min and Avg Air Temperature')
    axs[i].set_xlabel('Days')
    axs[i].set_ylabel('Temperature (C)')
    axs[i].legend()
    
    # Plot GP
    i=i+1
    axs[i].plot(gp, label='Growth Potential', color='magenta')
    axs[i].set_title('Growth Potential')
    axs[i].set_xlabel('Days')
    axs[i].set_ylabel('GP')
    axs[i].legend()
    
    
    # Plot TDD Accumulation
    i=i+1
    axs[i].plot(tdd_accumulation, label='TDD Accumulation', color='green')
    axs[i].set_title('TDD Accumulation')
    axs[i].set_xlabel('Days')
    axs[i].set_ylabel('TDD')
    axs[i].legend()
    
    # Plot growth
    i=i+1
    axs[i].plot(c_accumulation, label='Biomass Accumulation', color='blue')
    axs[i].set_title('Biomass Accumulation')
    axs[i].set_xlabel('Days')
    axs[i].set_ylabel('gC/m2')
    axs[i].legend()
    
    # Plot Chilling Hours
    i=i+1
    axs[i].plot(chilling_hours, label='Chilling Hours', color='purple')
    axs[i].set_title('Chilling Hours')
    axs[i].set_xlabel('Days')
    axs[i].set_ylabel('Hours')
    axs[i].legend()
    
    # Plot Is Dormant
    i=i+1
    axs[i].plot(is_dormant.astype(int), label='Is Dormant', color='red')
    axs[i].set_title('Is Dormant')
    axs[i].set_xlabel('Days')
    axs[i].set_ylabel('Is Dormant (0 or 1)')
    axs[i].legend()
    
    # Plot Consecutive Freezing Days
    i=i+1
    axs[i].plot(consecutive_freezing_days, label='Consecutive Freezing Days', color='cyan')
    axs[i].set_title('Consecutive Freezing Days')
    axs[i].set_xlabel('Days')
    axs[i].set_ylabel('Days')
    axs[i].legend()
    
    # Plot Consecutive Warming Days
    i=i+1
    axs[i].plot(consecutive_warming_days, label='Consecutive Warming Days', color='magenta')
    axs[i].set_title('Consecutive Warming Days')
    axs[i].set_xlabel('Days')
    axs[i].set_ylabel('Days')
    axs[i].legend()
    
    # Show the plot
    if title:
        fig.suptitle(title)
    plt.grid(True)
    if show:
        plt.show()
    # plt.show()
    plt.tight_layout()
    plt.close()
    return fig


def plot_growth_potential(df, optimum_growth_temp_c3, temp_variance_c3, 
                          optimum_growth_temp_c4, temp_variance_c4, title, figsize=(8, 4)):
    """
    Plot the Growth Potential (GP) for C3 and C4 grasses based on the given temperature data.
    df: DataFrame containing monthly average temperature data
    optimum_growth_temp_c3: Optimum Growth Temperature for C3 grasses in Celsius
    temp_variance_c3: Temperature variance for C3 grasses
    optimum_growth_temp_c4: Optimum Growth Temperature for C4 grasses in Celsius
    temp_variance_c4: Temperature variance for C4 grasses
    title: The title for the plot
    """
    # Calculate GP for C3 and C4 grasses
    df['GP_C3'] = df['tavg'].apply(lambda x: calculate_growth_potential(x, optimum_growth_temp_c3, temp_variance_c3, 'c3'))
    df['GP_C4'] = df['tavg'].apply(lambda x: calculate_growth_potential(x, optimum_growth_temp_c4, temp_variance_c4, 'c4'))
    
    # Plotting
    fig = plt.figure(figsize=figsize)
    plt.plot(df['year_month'].astype(str), df['GP_C3'], label='GP for C3 Grasses')
    plt.plot(df['year_month'].astype(str), df['GP_C4'], label='GP for C4 Grasses')
    plt.ylim(0, 1)
    plt.xlabel('Month')
    plt.ylabel('Growth Potential (GP)')
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.close()
    return fig


def plot_climate_and_growth_potential(lat, lon, 
                                      optimum_growth_temp_c3=20, temp_variance_c3=10, 
                                      optimum_growth_temp_c4=31, temp_variance_c4=12, 
                                      aggregation=2020, figsize=(8, 4), info=''):
    """
    Main function to fetch climate data, calculate GP, and plot the GP for C3 and C4 grasses.
    lat: Latitude of the location
    lon: Longitude of the location
    optimum_growth_temp_c3: Optimum Growth Temperature for C3 grasses in Celsius
    temp_variance_c3: Temperature variance for C3 grasses
    optimum_growth_temp_c4: Optimum Growth Temperature for C4 grasses in Celsius
    temp_variance_c4: Temperature variance for C4 grasses
    aggregation: 'average' to use the last 10 years of data, or a specific year to consider 3 consecutive years ending in that year
    """
    try:
        # Define the date range based on the aggregation parameter
        if aggregation == 'average':
            n_year = 10
            end_date = datetime.now()
            start_date = end_date - timedelta(days=n_year*365)  # Last 5 years
            title = f'Growth Potential for C3 and C4 Grasses (Last {n_year} Years, Averaged){info}'
        else:
            end_date = datetime(int(aggregation), 12, 31)
            start_date = datetime(int(aggregation) - 2, 1, 1)  # 3 years ending in the specified year
            title = f'Growth Potential for C3 and C4 Grasses ({int(aggregation)-2} - {int(aggregation)}){info}'
        
        # Fetch climate data
        print(start_date,end_date)
        try:
            _, climate_data = fetch_climate_data_regrow(lat, lon, start_date, end_date)
        except:
            _, climate_data = fetch_climate_data(lat, lon, start_date, end_date)
        
        # If aggregation is 'average', average the temperatures for each month across all years
        if aggregation == 'average':
            climate_data['month'] = climate_data['year_month'].dt.month
            climate_data = climate_data.groupby('month').agg({'tavg': 'mean'}).reset_index()
        
        # Plot the growth potential
        fig = plot_growth_potential(climate_data, optimum_growth_temp_c3, temp_variance_c3,
                              optimum_growth_temp_c4, temp_variance_c4, title, figsize)
        return fig
    
    except Exception as e:
        raise ValueError("Failed to obtain climate data or plot the growth potential. Error: " + str(e))

#### STREAMLIT PART #####    
st.set_page_config(layout='wide')

# Meta Information about the app
st.title("Growth Potential for Perennial Grasses")
st.subheader("Author: Trung Nguyen")
# st.text("Description: This app allows you to evaluate the growth potential of C3 and C4 perennial grasses based on temperature"
#         "\nYou can select a location on the map to get a plot of the growth potential for each type of grass.")

# Initialize last_clicked to None
last_clicked = None    

# Streamlit state
if 'plots' not in st.session_state:
    st.session_state['plots'] = []
if 'coords' not in st.session_state:
    st.session_state['coords'] = []
        
# Streamlit widgets to collect user inputs
st.sidebar.header('User Input Parameters')
aggregation = st.sidebar.number_input('Year:', value=2020)

# General parameters
species = st.sidebar.selectbox("Species:", options=["c3", "c4"])
max_n_month = st.sidebar.slider("Max monthly N uptake(g/m2/month):", 0.0, 10.0, 3.5)
c_n_ratio = st.sidebar.slider("C/N Ratio:", 0.0, 50.0, 20.0)

# Temperature thresholds
temp = st.sidebar.slider('Optimum Growth Temperature (°C):', 0, 50, 20)
var = st.sidebar.slider('Temperature Variance:', 0, 30, 10)
T_critical = st.sidebar.slider("Critical Temperature (°C):", -10.0, 10.0, -5.0)
T_base = st.sidebar.slider("Base Temperature (°C):", -10.0, 10.0, 0.0)
TDD_max = st.sidebar.slider("TDD Max:", 0.0, 5000.0, 2000.0)
tdd_min = st.sidebar.slider("TDD threshold for biomass accumulation:", 0.0, 5000.0, 10.0)

# Growth potential
gp_threshold = st.sidebar.slider("Growth Potential Threshold:", 0.0, 1.0, 0.1)
regrowth_reset_frac = st.sidebar.slider("Regrowth Reset Fraction:", 0.0, 1.0, 0.1)
TDD_max_reset_frac = st.sidebar.slider("TDD Max Reset Fraction:", 0.0, 1.0, 0.1)

# Chilling hours
min_chilling_hours = st.sidebar.slider("Min Chilling Hours:", 0, 500, 100)
chilling_temp_range = st.sidebar.slider("Chilling Temperature Range (0 - x):", 0, 10, (0, 7))

# Consecutive days
consecutive_freezing_days_threshold = st.sidebar.slider("Consecutive Freezing Days Threshold:", 0, 30, 7)
consecutive_warming_days_threshold = st.sidebar.slider("Consecutive Warming Days Threshold:", 0, 30, 7)
max_consecutive_day_tolerant = st.sidebar.slider("Max Consecutive Day Tolerant:", 0, 10, 3)


# st.subheader("Click on the map to plot growth potential")

# Create two columns using beta_columns
col1, col2 = st.columns(2)


# Create Folium map in the first column
with col1:
    
    # Create Folium map
    m = folium.Map(location=[50, -100], zoom_start=3,  width=400, height=600)
    
    # Remove the last marker if it exists
    if last_clicked:
        folium.Marker(last_clicked).remove()

    # Add click event support
    # map_data = st_folium(m, use_container_width=True)
    map_data = st_folium(m, width=400, height=600)

# Add click event support
if map_data:
    print(map_data)
    clicked_point = map_data.get('last_clicked')
    if clicked_point:
        latitude = clicked_point.get('lat')
        longitude = clicked_point.get('lng')
        st.write(f"Clicked coordinates: {latitude}, {longitude}")
        
        info = f"\n Lat,lon: {round(float(latitude),3)}, {round(float(longitude),3)} | Species: {species} | Opt T: {temp} | Var T: {var} | Year: {aggregation} \n"
        print(info)
        # Add a new marker
        with col1:
            folium.Marker([latitude, longitude]).add_to(m)
        
        # Update last_clicked
        last_clicked = [latitude, longitude]
        
        # Get date range based on aggregation
        end_date = datetime(int(aggregation), 12, 31)
        start_date = datetime(int(aggregation) - 2, 1, 1)  # 3 years ending in the specified year
        title = f'Growth Potential for C3 and C4 Grasses ({int(aggregation)-2} - {int(aggregation)})'
        
        # Get climate data
        weather_df, weather_month = fetch_climate_data(latitude, longitude, start_date, end_date, plot=True)
    
        # Generate the plot using the parameters and your function
        # fig = plot_climate_and_growth_potential(
        #     latitude, longitude,
        #     optimum_growth_temp_c3=temp_c3,
        #     temp_variance_c3=var_c3,
        #     optimum_growth_temp_c4=temp_c4,
        #     temp_variance_c4=var_c4,
        #     aggregation=aggregation,
        #     figsize=(16, 8),
        #     info=info
        # )
        
        tdd_accumulation_result_multiyear,\
        is_dormant_array, chilling_hours_array,\
        consecutive_freezing_days_array,\
        consecutive_warming_days_array, gp_array,\
        c_accumulation_array = dormancy_regrowth_logic(
                                                        weather_df['tmin'], 
                                                        weather_df['tavg'], 
                                                        species=species,
                                                        max_n_month=max_n_month,
                                                        c_n_ratio=c_n_ratio,
                                                        optimum_growth_temp=temp,
                                                        temp_variance=var,
                                                        T_critical=T_critical,
                                                        T_base=T_base,
                                                        TDD_max=TDD_max,
                                                        tdd_min=tdd_min,
                                                        gp_threshold=gp_threshold,
                                                        regrowth_reset_frac=regrowth_reset_frac,
                                                        TDD_max_reset_frac=TDD_max_reset_frac,
                                                        min_chilling_hours=min_chilling_hours, 
                                                        chilling_temp_range=chilling_temp_range,
                                                        consecutive_freezing_days_threshold=consecutive_freezing_days_threshold,
                                                        consecutive_warming_days_threshold=consecutive_warming_days_threshold,
                                                        max_consecutive_day_tolerant=max_consecutive_day_tolerant,
                                                        debug=False
                                                    )

        fig = visualize_dormancy_regrowth_logic(weather_df['tmin'],
                                            weather_df['tavg'],
                                            tdd_accumulation_result_multiyear, is_dormant_array,
                                            chilling_hours_array,
                                            consecutive_freezing_days_array,
                                            consecutive_warming_days_array,
                                            gp_array, c_accumulation_array,
                                            title=info)
        
        # Add the figure and coordinates to session state
        st.session_state['plots'].append(fig)
        st.session_state['coords'].append((round(float(latitude),3), round(float(longitude),3)))
        
        # Display the plot in Streamlit in the second column
        with col2:
            st.pyplot(fig)
            
            
# Display recorded plots
for i, (fig, coord) in enumerate(zip(st.session_state['plots'], st.session_state['coords'])):
    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        st.write(f"Lat/Lon: {coord[0]}/{coord[1]}")
    with col2:
        st.pyplot(fig)
    with col3:
        # Convert plot to a byte buffer
        buffer = BytesIO()
        fig.savefig(buffer, format="png")
        
        # Get the base64 encoding of the buffer
        plot_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Create a download link
        href = f'<a href="data:image/png;base64,{plot_base64}" download="plot_{i}.png">Download</a>'
        st.markdown(href, unsafe_allow_html=True)