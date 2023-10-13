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
        
        if monthly:
            # Resample to monthly frequency, taking the mean of each month
            df = df.resample('M').mean()
            df['year_month'] = df.index.to_period('M')
            
        if plot:
            # Plot line chart including average, minimum and maximum temperature
            df.plot(y=['tavg', 'tmin', 'tmax'])
            plt.show()
        return df
    else:
        print(f"Failed to retrieve data. HTTP Status Code: {response.status_code}")
        return None


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
        
        if monthly:
            # Resample to monthly frequency, taking the mean of each month
            df = df.resample('M').mean()
            df['year_month'] = df.index.to_period('M')
            
        if plot:
            # Plot line chart including average, minimum and maximum temperature
            df.plot(y=['tavg', 'tmin', 'tmax'])
            plt.show()
        return df
    else:
        print(f"Failed to retrieve data. HTTP Status Code: {response.status_code}")
        return None

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
    plt.xlabel('Month')
    plt.ylabel('Growth Potential (GP)')
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
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
            climate_data = fetch_climate_data_regrow(lat, lon, start_date, end_date)
        except:
            climate_data = fetch_climate_data(lat, lon, start_date, end_date)
        
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
st.text("Description: This app allows you to evaluate the growth potential of C3 and C4 perennial grasses based on temperature"
        "\nYou can select a location on the map to get a plot of the growth potential for each type of grass.")

# Initialize last_clicked to None
last_clicked = None    

# Streamlit state
if 'plots' not in st.session_state:
    st.session_state['plots'] = []
if 'coords' not in st.session_state:
    st.session_state['coords'] = []
        
# Streamlit widgets to collect user inputs
st.sidebar.header('User Input Parameters')
temp_c3 = st.sidebar.slider('Optimum temperature C3:', 0, 50, 20)
var_c3 = st.sidebar.slider('Temperature range C3:', 0, 50, 10)
temp_c4 = st.sidebar.slider('Optimum temperature  C4:', 0, 50, 31)
var_c4 = st.sidebar.slider('Temperature range C4:', 0, 50, 12)
aggregation = st.sidebar.number_input('Year:', value=2020)

st.subheader("Click on the map to plot growth potential")

# Create two columns using beta_columns
col1, col2 = st.columns(2)


# Create Folium map in the first column
with col1:
    
    # Create Folium map
    m = folium.Map(location=[50, -70], zoom_start=3, height=400, width=400)
    
    # Remove the last marker if it exists
    if last_clicked:
        folium.Marker(last_clicked).remove()

    # Add click event support
    # map_data = st_folium(m, use_container_width=True)
    map_data = st_folium(m, width=800, height=400)

# Add click event support
if map_data:
    print(map_data)
    clicked_point = map_data.get('last_clicked')
    if clicked_point:
        latitude = clicked_point.get('lat')
        longitude = clicked_point.get('lng')
        st.write(f"Clicked coordinates: {latitude}, {longitude}")
        
        info = f"\n Lat,lon: {latitude}, {longitude}"
        
        # Add a new marker
        with col1:
            folium.Marker([latitude, longitude]).add_to(m)
        
        # Update last_clicked
        last_clicked = [latitude, longitude]
    
        # Generate the plot using the parameters and your function
        fig = plot_climate_and_growth_potential(
            latitude, longitude,
            optimum_growth_temp_c3=temp_c3,
            temp_variance_c3=var_c3,
            optimum_growth_temp_c4=temp_c4,
            temp_variance_c4=var_c4,
            aggregation=aggregation,
            figsize=(16, 8),
            info=info
        )
        
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