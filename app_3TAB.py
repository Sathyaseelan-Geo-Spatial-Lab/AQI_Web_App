import os
from flask import Flask, jsonify, render_template, request
import ee
from google.oauth2 import service_account
from datetime import datetime, timedelta
import math  # Import math module for logarithmic calculations
import json
import traceback
from flask_caching import Cache
from functools import lru_cache
from collections import defaultdict
from datetime import datetime
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import requests
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import base64
import ssl
from flask_cors import CORS
import geopandas as gpd
from shapely.geometry import mapping

ssl._create_default_https_context = ssl._create_unverified_context

# Load environment variables
load_dotenv()
base_dir = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
CORS(app)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Constants for pollutant calculations
g = 9.82           # Acceleration due to gravity (m/s^2)
m_H2O = 0.01801528  # Molar mass of water vapor (kg/mol)
m_dry_air = 0.0289644  # Molar mass of dry air (kg/mol)
WINDY_API_KEY = "DHnqHp6YzeueWA6uhkK3cxT8USF5QsuX"

# Get credentials with error handling
try:
    creds = os.getenv("creds")  # or your actual env var name
    credentials_dict = json.loads(creds)  # ✅ FIXED HERE

    # Validate keys (optional)
    required_keys = ['type', 'project_id', 'private_key', 'client_email']
    for key in required_keys:
        if key not in credentials_dict:
            raise ValueError(f"Missing required key: {key}")

    print("Credentials loaded and validated.")

    credentials = service_account.Credentials.from_service_account_info(
        credentials_dict,
        scopes=['https://www.googleapis.com/auth/cloud-platform']
    )

    ee.Initialize(credentials)
    print("Earth Engine initialized successfully!")

except json.JSONDecodeError:
    print("Invalid JSON format in credentials")
except ValueError as e:
    print(f"Credentials validation error: {e}")
except Exception as e:
    print(f"Error initializing Google Earth Engine: {e}")

# Rest of your code remains the same...
def initialize_nltk():
    """Initialize NLTK by downloading required resources with error handling."""
    required_resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
    
    for resource in required_resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
                print(f"Successfully downloaded {resource}")
            except Exception as e:
                print(f"Error downloading {resource}: {str(e)}")
                # Continue with reduced functionality if download fails
                pass

# Call NLTK initialization at startup
initialize_nltk()

# Function to adjust units based on data region range
def adjust_units(min_value, max_value, base_unit):
    prefixes = {
        -12: '',
        -9: '',
        -6: '',
        -3: '',
        0: '',
        3: '',
        6: '',
        9: '',
        12: ''
    }

    abs_max = max(abs(min_value), abs(max_value))
    if abs_max == 0:
        exponent = 0
    else:
        exponent = int(math.floor(math.log10(abs_max)))
        exponent = (exponent // 3) * 3  # Round to nearest lower multiple of 3
        exponent = min(max(exponent, -12), 12)  # Limit exponent between -12 and 12

    scaling_factor = 10 ** (-exponent)
    prefix = prefixes.get(exponent, '')
    adjusted_unit = f"{prefix}{base_unit}"
    return scaling_factor, adjusted_unit

def interpolate_data_if_empty(pollutant, city_lat, city_lon, start_date, end_date, original_buffer):
    # If no data is available at the given buffer, we try larger buffers
    # This is a simple interpolation approach by expanding the search area until we find data.
    # We do not change any other logic, just attempt to find data in a larger radius and return mean.
    multipliers = [2, 5, 10]  # Try larger and larger radii
    for m in multipliers:
        new_buffer = original_buffer * m
        buffered_city_geometry = ee.Geometry.Point(city_lon, city_lat).buffer(new_buffer)
        if pollutant == 'PM10':
            filtered_collection = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_AER_AI') \
                .filterBounds(buffered_city_geometry) \
                .filterDate(start_date, end_date) \
                .select('absorbing_aerosol_index')
        elif pollutant == 'PM2.5':
            filtered_collection = ee.ImageCollection('MODIS/061/MCD19A2_GRANULES') \
                .filterBounds(buffered_city_geometry) \
                .filterDate(start_date, end_date) \
                .select('Optical_Depth_055')
        elif pollutant == 'NO2':
            filtered_collection = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_NO2') \
                .filterBounds(buffered_city_geometry) \
                .filterDate(start_date, end_date) \
                .select('NO2_column_number_density')
        elif pollutant == 'CO':
            filtered_collection = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_CO') \
                .filterBounds(buffered_city_geometry) \
                .filterDate(start_date, end_date) \
                .select(['CO_column_number_density']) #, 'H2O_column_number_density'
        elif pollutant == 'SO2':
            filtered_collection = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_SO2') \
                .filterBounds(buffered_city_geometry) \
                .filterDate(start_date, end_date) \
                .select('SO2_column_number_density')
        elif pollutant == 'O3':
            filtered_collection = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_O3') \
                .filterBounds(buffered_city_geometry) \
                .filterDate(start_date, end_date) \
                .select('O3_column_number_density')
        elif pollutant == 'HCHO':
            filtered_collection = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_HCHO') \
                .filterBounds(buffered_city_geometry) \
                .filterDate(start_date, end_date) \
                .select('tropospheric_HCHO_column_number_density')
        else:
            filtered_collection = None

        if filtered_collection and filtered_collection.size().getInfo() > 0:
            # We found data in a larger area. Take the mean and return it.
            def mask_negative_values(image):
                return image.updateMask(image.gte(0))
            filtered_collection = filtered_collection.map(mask_negative_values)
            mean_image = filtered_collection.mean().clip(buffered_city_geometry)
            # Return this mean image as "interpolated" data
            return mean_image, buffered_city_geometry
    # If no data found in any expanded search, return None
    return None, None

# Route for the home page
@app.route('/')
def index():
    return render_template('home_chart.html', windy_api_key=WINDY_API_KEY)

# Route for the about page
@app.route('/about/')
def about():
    return render_template('about.html')

# Route for the contact page
@app.route('/contact/')
def contact():
    return render_template('contact_us.html')

# API route to fetch pollutant data
@app.route('/api/get-pollutant-location', methods=['GET'])
def get_pollutant():
    try:
        # Retrieve parameters from the request
        city_lat = float(request.args.get('lat'))
        city_lon = float(request.args.get('lon'))
        buffer = request.args.get('buffer', default=50000, type=int)
        hml = request.args.get('hml', 'false').lower() == 'true'

        # Set default start and end dates (last 7 days)
        current_date = datetime.utcnow()
        default_end_date = current_date.strftime('%Y-%m-%dT%H:%M:%S')
        default_start_date = (current_date - timedelta(days=7)).strftime('%Y-%m-%dT%H:%M:%S')

        start_date = request.args.get('start_date', default_start_date)
        end_date = request.args.get('end_date', default_end_date)

        pollutant = request.args.get('pollutant', 'PM2.5')  # Default to PM2.5 if not specified

        if not city_lat or not city_lon:
            return jsonify({'error': 'Latitude and longitude are required parameters.'}), 400

        # Define a buffer around the specified point
        buffer_radius = buffer  # in meters
        buffered_city_geometry = ee.Geometry.Point(city_lon, city_lat).buffer(buffer_radius)
        bounds = buffered_city_geometry.bounds()

        geometry_data = {
            'geometry': buffered_city_geometry,
            'bounds': bounds
        }

        if pollutant == 'PM10':
            # Fetch the aerosol index data from Sentinel-5P
            filtered_collection = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_AER_AI') \
                .filterBounds(buffered_city_geometry) \
                .filterDate(start_date, end_date) \
                .select('absorbing_aerosol_index')

            if filtered_collection.size().getInfo() == 0:
                # Try interpolation
                mean_image, new_geom = interpolate_data_if_empty('PM10', city_lat, city_lon, start_date, end_date, buffer_radius)
                if mean_image is None:
                    return jsonify({'error': 'No PM10 data available for the specified parameters.'}), 404
                else:
                    aerosol_index_mean = mean_image
                    buffered_city_geometry = new_geom
            else:
                def mask_negative_values(image):
                    return image.updateMask(image.gte(0))
                filtered_collection = filtered_collection.map(mask_negative_values)
                aerosol_index_mean = filtered_collection.mean().clip(buffered_city_geometry)

            # Convert aerosol index to PM10 concentration
            PM10_mean = aerosol_index_mean.multiply(200).add(20)  # Adjust scaling factor and offset as needed
            pollutant_mean = PM10_mean.rename('PM10')

            stats = pollutant_mean.reduceRegion(
                reducer=ee.Reducer.minMax(),
                geometry=buffered_city_geometry,
                scale=1000,
                bestEffort=True
            ).getInfo()

            min_value = stats.get('PM10_min', None)
            max_value = stats.get('PM10_max', None)

            if min_value is None or max_value is None:
                return jsonify({'error': 'Could not calculate data range for PM10.'}), 500

            base_unit = 'µg/m³'
            scaling_factor, adjusted_unit = adjust_units(min_value, max_value, base_unit)

            pollutant_mean = pollutant_mean.multiply(scaling_factor)

            percentiles = pollutant_mean.reduceRegion(
                reducer=ee.Reducer.percentile([5, 95]),
                geometry=buffered_city_geometry,
                scale=1000,
                bestEffort=True
            ).getInfo()

            min_value = percentiles.get('PM10_p5', None)
            max_value = percentiles.get('PM10_p95', None)

            if min_value is None or max_value is None:
                return jsonify({'error': 'Could not calculate visualization parameters for PM10.'}), 500

            min_value = round(min_value, 2)
            max_value = round(max_value, 2)
            unit = adjusted_unit

        elif pollutant == 'PM2.5':
            # Fetch and process the PM2.5 data using the MODIS dataset
            filtered_collection = ee.ImageCollection('MODIS/061/MCD19A2_GRANULES') \
                .filterBounds(buffered_city_geometry) \
                .filterDate(start_date, end_date) \
                .select('Optical_Depth_055')

            if filtered_collection.size().getInfo() == 0:
                # Try interpolation
                mean_image, new_geom = interpolate_data_if_empty('PM2.5', city_lat, city_lon, start_date, end_date, buffer_radius)
                if mean_image is None:
                    return jsonify({'error': 'No PM2.5 data available for the specified parameters.'}), 404
                else:
                    PM2_5_mean = mean_image.multiply(206.91).add(41.181)
                    buffered_city_geometry = new_geom
            else:
                def mask_negative_values(image):
                    return image.updateMask(image.gte(0))
                filtered_collection = filtered_collection.map(mask_negative_values)
                PM2_5_mean = filtered_collection.mean().clip(buffered_city_geometry) \
                    .multiply(206.91).add(41.181)

            pollutant_mean = PM2_5_mean.rename('PM2_5')

            stats = pollutant_mean.reduceRegion(
                reducer=ee.Reducer.minMax(),
                geometry=buffered_city_geometry,
                scale=1000,
                bestEffort=True
            ).getInfo()

            min_value = stats.get('PM2_5_min', None)
            max_value = stats.get('PM2_5_max', None)

            if min_value is None or max_value is None:
                return jsonify({'error': 'Could not calculate data range for PM2.5.'}), 500

            base_unit = 'µg/m³'
            scaling_factor, adjusted_unit = adjust_units(min_value, max_value, base_unit)

            pollutant_mean = pollutant_mean.multiply(scaling_factor)

            percentiles = pollutant_mean.reduceRegion(
                reducer=ee.Reducer.percentile([5, 95]),
                geometry=buffered_city_geometry,
                scale=1000,
                bestEffort=True
            ).getInfo()

            min_value = percentiles.get('PM2_5_p5', None)
            max_value = percentiles.get('PM2_5_p95', None)

            if min_value is None or max_value is None:
                return jsonify({'error': 'Could not calculate visualization parameters for PM2.5.'}), 500

            min_value = round(min_value, 2)
            max_value = round(max_value, 2)
            unit = adjusted_unit

        elif pollutant == 'NO2':
            # Fetch and process the NO2 data from Sentinel-5P
            filtered_collection = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_NO2') \
                .filterBounds(buffered_city_geometry) \
                .filterDate(start_date, end_date) \
                .select('NO2_column_number_density')

            collection_size = filtered_collection.size().getInfo()
            print(f"NO2 collection size for the given parameters: {collection_size}")

            if collection_size == 0:
                # Try interpolation
                mean_image, new_geom = interpolate_data_if_empty('NO2', city_lat, city_lon, start_date, end_date, buffer_radius)
                if mean_image is None:
                    return jsonify({'error': 'No NO2 data available for the specified parameters.'}), 404
                else:
                    NO2_mean = mean_image
                    buffered_city_geometry = new_geom
            else:
                def mask_negative_values(image):
                    return image.updateMask(image.gte(0))
                filtered_collection = filtered_collection.map(mask_negative_values)
                NO2_mean = filtered_collection.mean().clip(buffered_city_geometry)
                print(NO2_mean)

            pollutant_mean = NO2_mean.rename('NO2')

            stats = pollutant_mean.reduceRegion(
                reducer=ee.Reducer.minMax(),
                geometry=buffered_city_geometry,
                scale=500,
                bestEffort=True
            ).getInfo()

            min_value = stats.get('NO2_min', None)
            max_value = stats.get('NO2_max', None)

            if min_value is None or max_value is None:
                return jsonify({'error': 'Could not calculate data range for NO2.'}), 500

            base_unit = 'mol/m²'
            scaling_factor, adjusted_unit = adjust_units(min_value, max_value, base_unit)

            pollutant_mean = pollutant_mean.multiply(scaling_factor)

            percentiles = pollutant_mean.reduceRegion(
                reducer=ee.Reducer.percentile([5, 95]),
                geometry=buffered_city_geometry,
                scale=500,
                bestEffort=True
            ).getInfo()

            min_value = percentiles.get('NO2_p5', None)
            max_value = percentiles.get('NO2_p95', None)

            if min_value is None or max_value is None:
                return jsonify({'error': 'Could not calculate visualization parameters for NO2.'}), 500

            # min_value = round(min_value, 6)
            # max_value = round(max_value, 6)

            if min_value == 0 and max_value == 0:
                return jsonify({'error': 'NO2 data is too low or not available for visualization in this area/date range.'}), 404

            unit = adjusted_unit
        
        
        elif pollutant == 'CH4':

    # ======================================================
    # Step 1: Fetch CH4 image collection
    # ======================================================

    filtered_collection = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_CH4') \
        .filterBounds(buffered_city_geometry) \
        .filterDate(start_date, end_date) \
        .select('CH4_column_volume_mixing_ratio_dry_air_bias_corrected')

    print("Filtered_completed")

    # ======================================================
    # Step 2: Check collection size
    # ======================================================

    collection_size = filtered_collection.size().getInfo()
    print(f"CH4 collection size for the given parameters: {collection_size}")

    if collection_size == 0:

        # Fallback to interpolation if no data
        mean_image, new_geom = interpolate_data_if_empty(
            'CH4',
            city_lat,
            city_lon,
            start_date,
            end_date,
            buffer_radius
        )

        if mean_image is None:
            return jsonify({
                'error': 'No CH4 data available for the specified parameters.'
            }), 404

        CH4_mean = mean_image
        buffered_city_geometry = new_geom

        # Rename band
        pollutant_mean = CH4_mean.rename('CH4')

    else:

        print("else_statement")

        # ======================================================
        # Step 3: Mask invalid CH4 values
        # ======================================================

        def mask_invalid_values(image):
            ch4 = image.select(
                'CH4_column_volume_mixing_ratio_dry_air_bias_corrected'
            )

            return image.updateMask(ch4.gte(0))

        filtered_collection = filtered_collection.map(mask_invalid_values)

        # ======================================================
        # Step 4: Compute mean and clip to area
        # ======================================================

        CH4_mean = filtered_collection.mean().clip(buffered_city_geometry)

        print("mean_calculation")

        # ======================================================
        # Step 5: Rename band
        # ======================================================

        pollutant_mean = CH4_mean.rename('CH4')

        # ======================================================
        # Step 6: Convert ppm → ppb
        # ======================================================

        pollutant_mean = pollutant_mean.multiply(1000)

        print("Converted CH4 from ppm to ppb")

        # ======================================================
        # Step 7: Validate unmasked pixels
        # ======================================================

        try:

            pixel_check = pollutant_mean.reduceRegion(
                reducer=ee.Reducer.count(),
                geometry=buffered_city_geometry,
                scale=500,
                bestEffort=True
            ).getInfo()

            print("Pixel count stats:", pixel_check)

            valid_pixel_count = pixel_check.get('CH4', 0)

            if valid_pixel_count == 0:

                print("No valid CH4 pixels found; attempting interpolation fallback...")

                mean_image, new_geom = interpolate_data_if_empty(
                    'CH4',
                    city_lat,
                    city_lon,
                    start_date,
                    end_date,
                    buffer_radius
                )

                if mean_image is None:
                    return jsonify({
                        'error': 'No valid CH4 data available even after interpolation.'
                    }), 404

                CH4_mean = mean_image
                buffered_city_geometry = new_geom

                pollutant_mean = CH4_mean.rename('CH4')

                # Convert fallback image also to ppb
                pollutant_mean = pollutant_mean.multiply(1000)

        except Exception as e:

            print("Error during pixel count check:", str(e))

            return jsonify({
                'error': 'Unexpected error when checking CH4 image validity.'
            }), 500

    # ======================================================
    # Step 8: Calculate min/max
    # ======================================================

    try:

        stats = pollutant_mean.reduceRegion(
            reducer=ee.Reducer.minMax(),
            geometry=buffered_city_geometry,
            scale=500,
            bestEffort=True
        ).getInfo()

        print("CH4 min/max stats:", stats)

        min_value = stats.get('CH4_min')
        max_value = stats.get('CH4_max')

        if min_value is None or max_value is None:
            raise ValueError("reduceRegion returned None for min/max.")

    except Exception as e:

        print("Error computing min/max stats for CH4:", str(e))

        return jsonify({
            'error': 'Could not calculate CH4 data range.'
        }), 500

    # ======================================================
    # Step 9: Units
    # ======================================================

    base_unit = 'ppb'

    scaling_factor, adjusted_unit = adjust_units(
        min_value,
        max_value,
        base_unit
    )

    print("Scaling factor:", scaling_factor)

    pollutant_mean = pollutant_mean.multiply(scaling_factor)

    # ======================================================
    # Step 10: Compute visualization percentiles
    # ======================================================

    try:

        percentiles = pollutant_mean.reduceRegion(
            reducer=ee.Reducer.percentile([5, 95]),
            geometry=buffered_city_geometry,
            scale=500,
            bestEffort=True
        ).getInfo()

        print("CH4 percentiles:", percentiles)

        min_value = percentiles.get('CH4_p5')
        max_value = percentiles.get('CH4_p95')

        if min_value is None or max_value is None:
            raise ValueError("reduceRegion returned None for percentiles.")

    except Exception as e:

        print("Error computing percentiles for CH4:", str(e))

        return jsonify({
            'error': 'Could not calculate visualization parameters for CH4.'
        }), 500

    # ======================================================
    # Step 11: Final validation
    # ======================================================

    min_value = round(min_value, 2)
    max_value = round(max_value, 2)

    if min_value == 0 and max_value == 0:

        return jsonify({
            'error': 'CH4 data is too low or unavailable for visualization.'
        }), 404

    unit = adjusted_unit
#### Final Corrected Output as per CPCB was Completed No Need To Change ########### 
        elif pollutant == 'CO':
            # Fetch and process the CO data from Sentinel-5P
            filtered_collection = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_CO') \
                .filterBounds(buffered_city_geometry) \
                .filterDate(start_date, end_date) \
                .select(['CO_column_number_density', 'H2O_column_number_density'])#

            surface_pressure_collection = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
                .filterBounds(buffered_city_geometry) \
                .filterDate(start_date, end_date) \
                .select('surface_pressure')
            t2_m_collection = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
                .filterBounds(buffered_city_geometry) \
                .filterDate(start_date, end_date) \
                .select('temperature_2m')
            if filtered_collection.size().getInfo() == 0 or surface_pressure_collection.size().getInfo() == 0:
                # Try interpolation
                mean_image, new_geom = interpolate_data_if_empty('CO', city_lat, city_lon, start_date, end_date, buffer_radius)
                if mean_image is None:
                    return jsonify({'error': 'No CO data available for the specified parameters.'}), 404
                else:
                    # We need H2O for CO calculation, try a larger approach for H2O too
                    filtered_collection_h2o = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_CO') \
                        .filterBounds(new_geom) \
                        .filterDate(start_date, end_date) \
                        .select(['H2O_column_number_density'])
                    if filtered_collection_h2o.size().getInfo() == 0:
                        return jsonify({'error': 'No CO data available for the specified parameters (H2O missing).'}), 404
                    def mask_negative_values(image):
                        return image.updateMask(image.gte(0))
                    filtered_collection_h2o = filtered_collection_h2o.map(mask_negative_values)
                    H2O_mean = filtered_collection_h2o.mean().clip(new_geom)

                    surface_pressure_collection_alt = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
                        .filterBounds(new_geom) \
                        .filterDate(start_date, end_date) \
                        .select('surface_pressure')

                    # if surface_pressure_collection_alt.size().getInfo() == 0:
                    #     return jsonify({'error': 'No CO data available for the specified parameters (Surface pressure missing).'}), 404
                    # surface_pressure_mean = surface_pressure_collection_alt.mean().clip(new_geom)
                    # print(surface_pressure_mean)
                    # CO_mean = mean_image
                    # buffered_city_geometry = new_geom
                    # # Calculate total column of dry air
                    # TC_dry_air = surface_pressure_mean.divide(g * m_dry_air).subtract(H2O_mean.multiply(m_H2O / m_dry_air))
                    # XCO_ppb = CO_mean.divide(TC_dry_air).multiply(1e9).rename('XCO_ppb')
                    # pollutant_mean = XCO_ppb

                    if surface_pressure_collection_alt.size().getInfo() == 0:
                        return jsonify({'error': 'No CO data available for the specified parameters.'}), 404
                    
                    CO_mean = mean_image
                    
                    M_CO = 28.01
                    MIXING_HEIGHT = 1000.0
                    
                    pollutant_mean = (
                        CO_mean
                            .multiply(M_CO)
                            .multiply(1000)
                            .divide(MIXING_HEIGHT)
                            .rename('CO_mg_m3')
                    )

                # end interpolation block
            else:
                def mask_negative_values(image):
                    return image.updateMask(image.gte(0))
                filtered_collection = filtered_collection.map(mask_negative_values)

                # Mean CO total column (mol/m²)
                CO_mean = filtered_collection.select(
                    'CO_column_number_density'
                ).mean().clip(buffered_city_geometry)
                
                # ------------------------------------------------------------------
                # Convert CO column density (mol/m²) → approximate CO concentration (mg/m³)
                # ------------------------------------------------------------------
                
                # Molecular weight of CO (g/mol)
                M_CO = 28.01
                
                # Assumed atmospheric mixing height (m)
                # Typical values:
                #   300–500 m : Stable/night-time
                #   800–1200 m: Typical daytime urban
                #   1500–2000 m: Well mixed atmosphere
                MIXING_HEIGHT = 1000.0
                
                # Conversion:
                # mg/m³ = (mol/m² × g/mol × 1000 mg/g) / mixing height (m)
                
                CO_mg_m3 = (
                    CO_mean
                        .multiply(M_CO)
                        .multiply(1000)
                        .divide(MIXING_HEIGHT)
                        .rename('CO_mg_m3')
                )
                
                print("CO concentration (mg/m³):", CO_mg_m3)
                
                pollutant_mean = CO_mg_m3

                # CO_mean = filtered_collection.select('CO_column_number_density').mean().clip(buffered_city_geometry)
                # H2O_mean = filtered_collection.select('H2O_column_number_density').mean().clip(buffered_city_geometry)
                # surface_pressure_mean = surface_pressure_collection.mean().clip(buffered_city_geometry)
                # t2m_mean = t2_m_collection.mean().clip(buffered_city_geometry)
                # print("Else_CO2_Pres:- ",surface_pressure_mean)
                # TC_dry_air = surface_pressure_mean.divide(g * m_dry_air).subtract(H2O_mean.multiply(m_H2O / m_dry_air))
                # print("TC_dry:- ",TC_dry_air)
                # #XCO_ppb = CO_mean.divide(TC_dry_air).multiply(1e9).rename('XCO_ppb')
                # M = 0.028010
                # print("M:- ",M)
                # R = 8.3145
                # print("R:- ",R)
                # a = CO_mean.divide(TC_dry_air)
                # print('a:- ',a)
                # b = (M/R)
                # print("B:- ",b)
                # c = surface_pressure_mean.divide(t2m_mean)
                # print("c:- ",c)
                # XCO_mg = a.multiply(b).multiply(c).multiply(1e6).rename('XCO_mg')
                # print("XCO_mg:- ",XCO_mg)
                # pollutant_mean = XCO_mg

            stats = pollutant_mean.reduceRegion(
                reducer=ee.Reducer.minMax(),
                geometry=buffered_city_geometry,
                scale=1000,
                bestEffort=True
            ).getInfo()
            print("stats:- ",stats)

            min_value = stats.get('XCO_mg_min', None)
            print("min:- ",min_value)
            max_value = stats.get('XCO_mg_max', None)

            if min_value is None or max_value is None:
                return jsonify({'error': 'Could not calculate data range for CO.'}), 500

            base_unit = 'mg/m³'
            min_value = round(min_value, 2)
            max_value = round(max_value, 2)
            unit = base_unit

        elif pollutant == 'SO2':
            # Fetch and process the SO2 data from Sentinel-5P
            filtered_collection = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_SO2') \
                .filterBounds(buffered_city_geometry) \
                .filterDate(start_date, end_date) \
                .select('SO2_column_number_density')

            if filtered_collection.size().getInfo() == 0:
                # Try interpolation
                mean_image, new_geom = interpolate_data_if_empty('SO2', city_lat, city_lon, start_date, end_date, buffer_radius)
                if mean_image is None:
                    return jsonify({'error': 'No SO2 data available for the specified parameters.'}), 404
                else:
                    SO2_mean = mean_image
                    buffered_city_geometry = new_geom
            else:
                def mask_negative_values(image):
                    return image.updateMask(image.gte(0))
                filtered_collection = filtered_collection.map(mask_negative_values)
                SO2_mean = filtered_collection.mean().clip(buffered_city_geometry)

            pollutant_mean = SO2_mean.rename('SO2')

            stats = pollutant_mean.reduceRegion(
                reducer=ee.Reducer.minMax(),
                geometry=buffered_city_geometry,
                scale=1000,
                bestEffort=True
            ).getInfo()

            min_value = stats.get('SO2_min', None)
            max_value = stats.get('SO2_max', None)

            if min_value is None or max_value is None:
                return jsonify({'error': 'Could not calculate data range for SO2.'}), 500

            base_unit = 'mol/m²'
            scaling_factor, adjusted_unit = adjust_units(min_value, max_value, base_unit)

            pollutant_mean = pollutant_mean.multiply(scaling_factor)

            percentiles = pollutant_mean.reduceRegion(
                reducer=ee.Reducer.percentile([5, 95]),
                geometry=buffered_city_geometry,
                scale=1000,
                bestEffort=True
            ).getInfo()

            min_value = percentiles.get('SO2_p5', None)
            max_value = percentiles.get('SO2_p95', None)

            if min_value is None or max_value is None:
                return jsonify({'error': 'Could not calculate visualization parameters for SO2.'}), 500

            min_value = round(min_value, 2)
            max_value = round(max_value, 2)

            if min_value == 0 and max_value == 0:
                return jsonify({'error': 'SO2 data is too low or not available for visualization in this area/date range.'}), 404

            unit = adjusted_unit

        elif pollutant == 'O3':
            # Fetch and process the O3 data from Sentinel-5P
            filtered_collection = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_O3') \
                .filterBounds(buffered_city_geometry) \
                .filterDate(start_date, end_date) \
                .select('O3_column_number_density')

            if filtered_collection.size().getInfo() == 0:
                # Try interpolation
                mean_image, new_geom = interpolate_data_if_empty('O3', city_lat, city_lon, start_date, end_date, buffer_radius)
                if mean_image is None:
                    return jsonify({'error': 'No O3 data available for the specified parameters.'}), 404
                else:
                    O3_mean = mean_image
                    buffered_city_geometry = new_geom
            else:
                def mask_negative_values(image):
                    return image.updateMask(image.gte(0))
                filtered_collection = filtered_collection.map(mask_negative_values)
                O3_mean = filtered_collection.mean().clip(buffered_city_geometry)

            pollutant_mean = O3_mean.rename('O3')

            stats = pollutant_mean.reduceRegion(
                reducer=ee.Reducer.minMax(),
                geometry=buffered_city_geometry,
                scale=1000,
                bestEffort=True
            ).getInfo()

            min_value = stats.get('O3_min', None)
            max_value = stats.get('O3_max', None)

            if min_value is None or max_value is None:
                return jsonify({'error': 'Could not calculate data range for O3.'}), 500

            base_unit = 'mol/m²'
            scaling_factor, adjusted_unit = adjust_units(min_value, max_value, base_unit)

            pollutant_mean = pollutant_mean.multiply(scaling_factor)

            percentiles = pollutant_mean.reduceRegion(
                reducer=ee.Reducer.percentile([5, 95]),
                geometry=buffered_city_geometry,
                scale=1000,
                bestEffort=True
            ).getInfo()

            min_value = percentiles.get('O3_p5', None)
            max_value = percentiles.get('O3_p95', None)

            if min_value is None or max_value is None:
                return jsonify({'error': 'Could not calculate visualization parameters for O3.'}), 500

            min_value = round(min_value, 2)
            max_value = round(max_value, 2)

            if min_value == 0 and max_value == 0:
                return jsonify({'error': 'O3 data is too low or not available for visualization in this area/date range.'}), 404

            unit = adjusted_unit

        elif pollutant == 'HCHO':
            # Fetch and process the HCHO data from Sentinel-5P
            filtered_collection = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_HCHO') \
                .filterBounds(buffered_city_geometry) \
                .filterDate(start_date, end_date) \
                .select('tropospheric_HCHO_column_number_density')

            if filtered_collection.size().getInfo() == 0:
                # Try interpolation
                mean_image, new_geom = interpolate_data_if_empty('HCHO', city_lat, city_lon, start_date, end_date, buffer_radius)
                if mean_image is None:
                    return jsonify({'error': 'No HCHO data available for the specified parameters.'}), 404
                else:
                    HCHO_mean = mean_image
                    buffered_city_geometry = new_geom
            else:
                def mask_negative_values(image):
                    return image.updateMask(image.gte(0))
                filtered_collection = filtered_collection.map(mask_negative_values)
                HCHO_mean = filtered_collection.mean().clip(buffered_city_geometry)

            pollutant_mean = HCHO_mean.rename('HCHO')

            stats = pollutant_mean.reduceRegion(
                reducer=ee.Reducer.minMax(),
                geometry=buffered_city_geometry,
                scale=1000,
                bestEffort=True
            ).getInfo()

            min_value = stats.get('HCHO_min', None)
            max_value = stats.get('HCHO_max', None)

            if min_value is None or max_value is None:
                return jsonify({'error': 'Could not calculate data range for HCHO.'}), 500

            base_unit = 'mol/m²'
            scaling_factor, adjusted_unit = adjust_units(min_value, max_value, base_unit)

            pollutant_mean = pollutant_mean.multiply(scaling_factor)

            percentiles = pollutant_mean.reduceRegion(
                reducer=ee.Reducer.percentile([5, 95]),
                geometry=buffered_city_geometry,
                scale=1000,
                bestEffort=True
            ).getInfo()

            min_value = percentiles.get('HCHO_p5', None)
            max_value = percentiles.get('HCHO_p95', None)

            if min_value is None or max_value is None:
                return jsonify({'error': 'Could not calculate visualization parameters for HCHO.'}), 500

            min_value = round(min_value, 2)
            max_value = round(max_value, 2)

            if min_value == 0 and max_value == 0:
                return jsonify({'error': 'HCHO data is too low or not available for visualization in this area/date range.'}), 404

            unit = adjusted_unit

        else:
            return jsonify({'error': f"Unsupported pollutant: {pollutant}"}), 400

        if min_value == max_value:
            min_value -= 0.1 * abs(min_value) or 0.1
            max_value += 0.1 * abs(max_value) or 0.1

        buffer_range = abs(max_value - min_value) * 0.1
        if hml:
            vis_params = {
                'min': min_value,
                'max': max_value,
                'palette': ['blue', 'yellow', 'red'],
            }
            legend_labels = ['Low', 'Medium', 'High']
        else:
            buffer_range = abs(max_value - min_value) * 0.1
            vis_params = {
                'min': min_value - buffer_range,
                'max': max_value + buffer_range,
                'palette': ['blue', 'cyan', 'green', 'yellow', 'red']
            }
            legend_labels = None
        time_series = compute_timeseries_mean_3(geometry_data, pollutant, start_date, end_date)
        map_id = pollutant_mean.getMapId(vis_params)
        tile_url = map_id['tile_fetcher'].url_format

        min_value_sci = f"{min_value:.2e}"
        max_value_sci = f"{max_value:.2e}"

        return jsonify({
                        'tile_url': tile_url,
                        'min': min_value_sci,
                        'max': max_value_sci,
                        'min_raw': min_value,
                        'max_raw': max_value,
                        'unit': unit,
                        'legend_labels': legend_labels,
                        'time_series': time_series
                    })

    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

POLLUTANT_CONFIGS = {
    'PM2.5': {
        'collection': 'MODIS/061/MCD19A2_GRANULES',
        'band': 'Optical_Depth_055',
        'scale_factor': 0.08,#0.20691 
        'offset': 5,
        'unit': 'µg/m³'
    },
    'PM10': {
        'collection': 'COPERNICUS/S5P/NRTI/L3_AER_AI',
        'band': 'absorbing_aerosol_index',
        'scale_factor': 200, #with code orig val is 50 # Conversion factor for PM10
        'offset': 20,        # Offset for PM10
        'unit': 'μg/m³'
    },
    'NO2': {
        'collection': 'COPERNICUS/S5P/NRTI/L3_NO2',
        'band': 'NO2_column_number_density',
        'unit': 'μg/m³',
        'scale_factor':50,
        'offset':0,
    },
    'CO': {
        'collection': 'COPERNICUS/S5P/NRTI/L3_CO',
        'band': 'CO_column_number_density',
        'unit': 'mg/m³',
        'scale_factor':11.204,#previous value is 3000,#present value H = 1000m 
        'offset':0
    },
    'SO2': {
        'collection': 'COPERNICUS/S5P/NRTI/L3_SO2',
        'band': 'SO2_column_number_density',
        'unit': 'μg/m³', # 'μg/m3' scale factor is 64066 1 mol/m2=64,066 μg/m3
        'scale_factor':64, #1000000
        'offset':5

    },
    'O3': {
        'collection': 'COPERNICUS/S5P/NRTI/L3_O3',
        'band': 'O3_column_number_density',
        'unit': 'μg/m³',
        'scale_factor':48,
        'offset':17.5
    },
    'HCHO': {
        'collection': 'COPERNICUS/S5P/NRTI/L3_HCHO',
        'band': 'tropospheric_HCHO_column_number_density',
        'scale_factor':15,
        'offset':0,
        'unit': 'μg/m³'

    },
    'CH4':
    {
       'collection': "COPERNICUS/S5P/OFFL/L3_CH4",
        'band': 'CH4_column_volume_mixing_ratio_dry_air_bias_corrected',
        'scale_factor':1,
        'offset':0,
        'unit': 'ppb' 
    }
}

@cache.memoize(timeout=3600)

def get_optimized_geometry(geojson_path, simplify_error=1000):
    """Load and optimize geometry from GeoJSON file."""
    with open(geojson_path, 'r') as f:
        geojson_data = json.load(f)
    geometry = ee.Geometry(geojson_data['features'][0]['geometry'])
    simplified_geom = geometry.simplify(maxError=simplify_error)
    bounds = simplified_geom.bounds()
    #print(bounds)
    return {
        'geometry': simplified_geom,
        'bounds': bounds
    }



# Dictionary to store all processed pollutant data
pollutant_data_dict = {}

# Process Pollutant Function New
###############################################################################################################
def process_pollutant_data(geometry_data, pollutant, start_date, end_date, scale=1000, allow_interpolation=False):
    if pollutant not in POLLUTANT_CONFIGS:
        raise ValueError(f'Unsupported pollutant: {pollutant}')

    config = POLLUTANT_CONFIGS[pollutant]
    bounds = geometry_data['bounds']
    geometry = geometry_data['geometry']

    # Filter collection
    collection = ee.ImageCollection(config['collection']) \
        .filterBounds(bounds) \
        .filterDate(start_date, end_date) \
        .select(config['band'])

    # Check for empty collection and fallback if allowed
    if collection.size().getInfo() == 0 and allow_interpolation:
        # Get center coordinates of the geometry for interpolation
        centroid = geometry.centroid()
        lon = centroid.coordinates().getInfo()[0]
        lat = centroid.coordinates().getInfo()[1]

        # Use a fallback interpolation method (user-defined elsewhere)
        interpolated_image, interpolated_geom = interpolate_data_if_empty(
            pollutant, lat, lon, start_date, end_date, buffer_radius=10000
        )

        if interpolated_image is None:
            raise ValueError(f"No data found for {pollutant}, and interpolation failed.")

        mean_image = interpolated_image
        geometry = interpolated_geom
    else:
        def mask_negative_values(image):
            return image.updateMask(image.gte(0))
        collection = collection.map(mask_negative_values)

        mean_image = collection.mean()

        # Apply scale factor and offset if configured
        if 'scale_factor' in config:
            mean_image = mean_image.multiply(config['scale_factor'])
            if 'offset' in config:
                mean_image = mean_image.add(config['offset'])

    # Rename bands appropriately
    if pollutant == 'PM2.5':
        masked_mean = mean_image.clip(geometry).rename('PM2_5')
    else:
        masked_mean = mean_image.clip(geometry).rename(pollutant)

    # Compute stats for visualization
    stats = masked_mean.reduceRegion(
        reducer=ee.Reducer.percentile([5, 95]),
        geometry=geometry,
        scale=scale,
        maxPixels=1e8,
        bestEffort=True
    ).getInfo()

    return masked_mean, stats, config['unit']

#For non AQI calculation
def compute_timeseries_mean(geometry_data, pollutant, start_date, end_date, scale=1000):
    
    if pollutant not in POLLUTANT_CONFIGS:
        raise ValueError(f'Unsupported pollutant: {pollutant}')

    config = POLLUTANT_CONFIGS[pollutant]
    bounds = geometry_data['bounds']
    print("pollutant")
    geometry = geometry_data['geometry']

    collection = ee.ImageCollection(config['collection']) \
        .filterBounds(bounds) \
        .filterDate(start_date, end_date) \
        .select(config['band'])
    print("collection")
    def mask_negative_values(image):
        return image.updateMask(image.gte(0))

    collection = collection.map(mask_negative_values)

    band_name = config['band']
    print(band_name)

    def image_to_feature(image):
        img = image
        if 'scale_factor' in config:
            img = img.multiply(config['scale_factor'])
            if 'offset' in config:
                img = img.add(config['offset'])

        img_clipped = img.clip(geometry)

        mean_dict = img_clipped.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=scale,
            maxPixels=1e8,
            bestEffort=True
        )

        mean_val = mean_dict.get(band_name)

        return ee.Feature(None, {
            'date': image.date().format('YYYY-MM-dd'),
            'value': mean_val
        })

    feature_collection = collection.map(image_to_feature).filter(ee.Filter.notNull(['value']))

    # Get client-side list of features
    features_list = feature_collection.toList(feature_collection.size()).getInfo()
    
    results = []
    for feature in features_list:
        props = feature.get('properties', {})
        d = props.get('date')
        v = props.get('value')
        if v is not None:
            results.append({
                'date': d,
                'value': v,
                'band': 'PM2_5' if pollutant == 'PM2.5' else pollutant
            })
    print("results crossed")
    return results

def compute_timeseries_mean_3(geometry_data, pollutant, start_date, end_date, scale=1000):

    if pollutant not in POLLUTANT_CONFIGS:
        raise ValueError(f'Unsupported pollutant: {pollutant}')

    config = POLLUTANT_CONFIGS[pollutant]
    bounds = geometry_data['bounds']
    geometry = geometry_data['geometry']

    def mask_negative_values(image):
        return image.updateMask(image.gte(0))

    results = []

    # ✅ Special Case: CO — Use Ideal Gas Law to compute mass concentration in mg/m³
    if pollutant == 'CO':
        # Load all required collections
        co_collection = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_CO') \
                .filterBounds(bounds) \
                .filterDate(start_date, end_date) \
                .select(['CO_column_number_density', 'H2O_column_number_density'])\
            .map(mask_negative_values)

        pressure_collection = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
                .filterBounds(bounds) \
                .filterDate(start_date, end_date) \
                .select('surface_pressure')

        temp_collection = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
                .filterBounds(bounds) \
                .filterDate(start_date, end_date) \
                .select('temperature_2m')

        # Constants
        g = 9.80665
        m_dry_air = 0.0289644
        m_H2O = 0.01801528
        M = 0.02801
        R = 8.3145

        def compute_daily_XCO(date):
            date = ee.Date(date)

            # Load daily images
            co_img = co_collection.select('CO_column_number_density') \
                .filterDate(date, date.advance(1, 'day')).mean()

            h2o_img = co_collection.select('H2O_column_number_density') \
                .filterDate(date, date.advance(1, 'day')).mean()

            pressure_img = pressure_collection.filterDate(date, date.advance(1, 'day')).mean()
            temp_img = temp_collection.filterDate(date, date.advance(1, 'day')).mean()

            # Check if all required images have at least one band
            valid = (
                co_img.bandNames().size().gt(0)
                .And(h2o_img.bandNames().size().gt(0))
                .And(pressure_img.bandNames().size().gt(0))
                .And(temp_img.bandNames().size().gt(0))
            )

            def compute_feature():
                TC_dry = pressure_img.divide(g * m_dry_air).subtract(
                    h2o_img.multiply(m_H2O / m_dry_air)
                )
                a = co_img.divide(TC_dry)
                b = M / R
                c = pressure_img.divide(temp_img)
                XCO_mg = a.multiply(b).multiply(c).multiply(1e6).rename('XCO_mg')

                mean_dict = XCO_mg.clip(geometry).reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=geometry,
                    scale=scale,
                    maxPixels=1e8,
                    bestEffort=True
                )

                return ee.Feature(None, {
                    'date': date.format('YYYY-MM-dd'),
                    'value': mean_dict.get('XCO_mg')
                })

            return ee.Algorithms.If(valid, compute_feature(), None)
        dates = co_collection.aggregate_array('system:time_start')
        date_list = ee.List(dates).map(lambda d: ee.Date(d))

        # Map and remove None values
        features = date_list.map(compute_daily_XCO)
        feature_collection = ee.FeatureCollection(features).filter(ee.Filter.notNull(['value']))

        # Optional: check if result is empty
        if feature_collection.size().getInfo() == 0:
            raise ValueError("No valid CO + H2O + pressure + temp data found in selected range.")

    else:
        # ✅ Default Case: Other pollutants
        collection = ee.ImageCollection(config['collection']) \
            .filterBounds(bounds) \
            .filterDate(start_date, end_date) \
            .select(config['band']) \
            .map(mask_negative_values)

        band_name = config['band']

        def image_to_feature(image):
            img = image
            if 'scale_factor' in config:
                img = img.multiply(config['scale_factor'])
                if 'offset' in config:
                    img = img.add(config['offset'])

            mean_dict = img.clip(geometry).reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=scale,
                maxPixels=1e8,
                bestEffort=True
            )

            mean_val = mean_dict.get(band_name)

            return ee.Feature(None, {
                'date': image.date().format('YYYY-MM-dd'),
                'value': mean_val
            })

        feature_collection = collection.map(image_to_feature).filter(ee.Filter.notNull(['value']))

    # 🔁 Convert to client-side list
    features_list = feature_collection.toList(feature_collection.size()).getInfo()

    for feature in features_list:
        props = feature.get('properties', {})
        d = props.get('date')
        v = props.get('value')
        if v is not None:
            results.append({
                'date': d,
                'value': v,
                'band': 'CO' if pollutant == 'CO' else ('PM2_5' if pollutant == 'PM2.5' else pollutant)
            })

    return results



"""def compute_timeseries_mean(geometry_data, pollutant, start_date, end_date, scale=1000):
    if pollutant not in POLLUTANT_CONFIGS:
        raise ValueError(f'Unsupported pollutant: {pollutant}')

    config = POLLUTANT_CONFIGS[pollutant]
    bounds = geometry_data['bounds']
    geometry = geometry_data['geometry']

    def mask_negative_values(image):
        return image.updateMask(image.gte(0))

    results = []

    # ✅ Special Case: CO — Use Ideal Gas Law to compute mass concentration in mg/m³
    if pollutant == 'CO':
        # Load all required collections
        co_collection = ee.ImageCollection(config['collection']) \
            .filterDate(start_date, end_date) \
            .filterBounds(bounds) \
            .select(['CO_column_number_density', 'H2O_column_number_density']) \
            .map(mask_negative_values)

        pressure_collection = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
                .filterBounds(bounds) \
                .filterDate(start_date, end_date) \
                .select('surface_pressure')

        temp_collection = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
                .filterBounds(bounds) \
                .filterDate(start_date, end_date) \
                .select('temperature_2m')
        if co_collection.size().getInfo() == 0 or pressure_collection.size().getInfo() == 0 or temp_collection.size().getInfo() == 0:
                # Try interpolation
                mean_image, new_geom = interpolate_data_if_empty('CO', city_lat, city_lon, start_date, end_date, buffer_radius)
                if mean_image is None:
                    return jsonify({'error': 'No CO data available for the specified parameters.'}), 404

        # Constants
        g = 9.80665
        m_dry_air = 0.0289644
        m_H2O = 0.01801528
        M = 0.02801
        R = 8.3145

        def compute_daily_XCO(date):
            date = ee.Date(date)
            co_img = co_collection.select('CO_column_number_density').filterDate(date, date.advance(1, 'day')).mean()
            h2o_img = co_collection.select('H2O_column_number_density').filterDate(date, date.advance(1, 'day')).mean()
            pressure_img = pressure_collection.filterDate(date, date.advance(1, 'day')).mean()
            temp_img = temp_collection.filterDate(date, date.advance(1, 'day')).mean()

            TC_dry = pressure_img.divide(g * m_dry_air).subtract(h2o_img.multiply(m_H2O / m_dry_air))
            a = co_img.divide(TC_dry)
            b = M / R
            c = pressure_img.divide(temp_img)
            XCO_mg = a.multiply(b).multiply(c).multiply(1e6).rename('XCO_mg')

            mean_dict = XCO_mg.clip(geometry).reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=scale,
                maxPixels=1e8,
                bestEffort=True
            )

            return ee.Feature(None, {
                'date': date.format('YYYY-MM-dd'),
                'value': mean_dict.get('XCO_mg')
            })

        # Build list of dates from CO collection
        dates = co_collection.aggregate_array('system:time_start')
        date_list = ee.List(dates).map(lambda d: ee.Date(d))

        # Map over each date to compute daily CO
        feature_collection = ee.FeatureCollection(date_list.map(compute_daily_XCO)) \
            .filter(ee.Filter.notNull(['value']))

    else:
        # ✅ Default Case: Other pollutants
        collection = ee.ImageCollection(config['collection']) \
            .filterBounds(bounds) \
            .filterDate(start_date, end_date) \
            .select(config['band']) \
            .map(mask_negative_values)

        band_name = config['band']

        def image_to_feature(image):
            img = image
            if 'scale_factor' in config:
                img = img.multiply(config['scale_factor'])
                if 'offset' in config:
                    img = img.add(config['offset'])

            mean_dict = img.clip(geometry).reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=scale,
                maxPixels=1e8,
                bestEffort=True
            )

            mean_val = mean_dict.get(band_name)

            return ee.Feature(None, {
                'date': image.date().format('YYYY-MM-dd'),
                'value': mean_val
            })

        feature_collection = collection.map(image_to_feature).filter(ee.Filter.notNull(['value']))

    # 🔁 Convert to client-side list
    features_list = feature_collection.toList(feature_collection.size()).getInfo()

    for feature in features_list:
        props = feature.get('properties', {})
        d = props.get('date')
        v = props.get('value')
        if v is not None:
            results.append({
                'date': d,
                'value': v,
                'band': 'CO' if pollutant == 'CO' else ('PM2_5' if pollutant == 'PM2.5' else pollutant)
            })

    return results
"""

# Functions to get the city state or the buffer geometries to calculate the regional timeseries mean 
#############################################################################################################################
def compute_timeseries_mean_2(geometry_data, pollutant, start_date, end_date, scale=1000):
    try:
        print(f"Computing time series for pollutant={pollutant} from {start_date} to {end_date}")

        if pollutant not in POLLUTANT_CONFIGS:
            raise ValueError(f'Unsupported pollutant: {pollutant}')

        config = POLLUTANT_CONFIGS[pollutant]
        bounds = geometry_data['bounds']
        geometry = geometry_data['geometry']

        collection = ee.ImageCollection(config['collection']) \
            .filterBounds(bounds) \
            .filterDate(start_date, end_date) \
            .select(config['band'])
        print('collections crossed')
        def mask_negative_values(image):
            return image.updateMask(image.gte(0))

        collection = collection.map(mask_negative_values)

        band_name = config['band']
        print("band name crossed")
        def image_to_feature(image):
            img = image
            if 'scale_factor' in config:
                img = img.multiply(config['scale_factor'])
                if 'offset' in config:
                    img = img.add(config['offset'])

            img_clipped = img.clip(geometry)

            mean_dict = img_clipped.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=scale,
                maxPixels=1e8,
                bestEffort=True
            )

            mean_val = mean_dict.get(band_name)

            return ee.Feature(None, {
                'date': image.date().format('YYYY-MM-dd'),
                'value': mean_val
            })

        feature_collection = collection.map(image_to_feature).filter(ee.Filter.notNull(['value']))
        
        # Get client-side list of features
        features_list = feature_collection.toList(feature_collection.size()).getInfo()

        results = []
        for feature in features_list:
            props = feature.get('properties', {})
            d = props.get('date')
            v = props.get('value')
            if v is not None:
                results.append({
                    'date': d,
                    'value': v,
                    'band': 'PM2_5' if pollutant == 'PM2.5' else pollutant
                })
        grouped = defaultdict(list)
        for item in results:
            grouped[item['date']].append(item['value'])

        averaged_results = []
        for date, values in grouped.items():
            averaged_results.append({
                'date': date,
                'value': float(np.mean(values)),
                'band': results[0]['band']  # assuming all same pollutant
            })

        # Optional: sort by date
        averaged_results.sort(key=lambda x: x['date'])

        return averaged_results
    except:
        return 

    
# API Route for the getting the City based pollutant data and display their legends
#############################################################################################################################
@app.route('/api/get-pollutant-city', methods=['GET'])
def get_pollutant_city():
    try:
        city = request.args.get('city')
        print(city)
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        pollutant = request.args.get('pollutant')
        print(pollutant)
        hml = request.args.get('hml', 'false').lower() == 'true'

        if not all([city, start_date, end_date, pollutant]):
            return jsonify({'error': 'Missing required parameters'}), 400

        city_data = get_optimized_geometry(
            f"static/city/{city.upper()}.geojson",
            simplify_error=100
        )

        masked_mean, stats, unit = process_pollutant_data(
            city_data, pollutant, start_date, end_date, scale=1000
        )
        #print(unit)
        # ---------------------------------------
        # Compute time series of regional means
        # ---------------------------------------
        time_series = compute_timeseries_mean(city_data, pollutant, start_date, end_date)
        print('timeseries')
        band_name = 'PM2_5' if pollutant == 'PM2.5' else pollutant
        try:
            min_value = float(stats[f'{band_name}_p5'])
            max_value = float(stats[f'{band_name}_p95'])
        except (KeyError, TypeError, ValueError):
            return jsonify({'error': f'Could not calculate valid stats for {pollutant}.'}), 500

        if min_value == max_value:
            min_value -= 0.1 * abs(min_value) or 0.1
            max_value += 0.1 * abs(max_value) or 0.1

        if hml:
            vis_params = {
                'min': min_value,
                'max': max_value,
                'palette': ['blue', 'yellow', 'red']
            }
            legend_labels = ['Low', 'Medium', 'High']
        else:
            buffer_range = abs(max_value - min_value) * 0.1
            vis_params = {
                'min': min_value - buffer_range,
                'max': max_value + buffer_range,
                'palette': ['blue', 'cyan', 'green', 'yellow', 'red']
            }
            legend_labels = None

        map_id = masked_mean.getMapId(vis_params)
        
        print(min_value)
        return jsonify({
            'tile_url': map_id['tile_fetcher'].url_format,
            'min': f"{min_value:.2e}",
            'max': f"{max_value:.2e}",
            'min_raw': min_value,
            'max_raw': max_value,
            'unit': unit,
            'legend_labels': legend_labels,
            'time_series': time_series
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# API Route for the getting the State based pollutant data and display their legends
########################################################################################    , fontdict=fontdict
@app.route('/api/get-pollutant-state', methods=['GET'])
def get_pollutant_state():
    try:
        state = request.args.get('state')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        pollutant = request.args.get('pollutant')
        hml = request.args.get('hml', 'false').lower() == 'true'

        if not all([state, start_date, end_date, pollutant]):
            return jsonify({'error': 'Missing required parameters'}), 400

        state_data = get_optimized_geometry(f"static/state/{state}.geojson")

        masked_mean, stats, unit = process_pollutant_data(
            state_data, pollutant, start_date, end_date
        )
        
         # ---------------------------------------
        # Compute time series of regional means
        # ---------------------------------------
        time_series = compute_timeseries_mean(state_data, pollutant, start_date, end_date)
    

        band_name = 'PM2_5' if pollutant == 'PM2.5' else pollutant
        try:
            min_value = float(stats[f'{band_name}_p5'])
            max_value = float(stats[f'{band_name}_p95'])
        except (KeyError, TypeError, ValueError):
            return jsonify({'error': f'Could not calculate valid stats for {pollutant}.'}), 500

        if min_value == max_value:
            min_value -= 0.1 * abs(min_value) or 0.1
            max_value += 0.1 * abs(max_value) or 0.1

        if hml:
            vis_params = {
                'min': min_value,
                'max': max_value,
                'palette': ['blue', 'yellow', 'red']
            }
            legend_labels = ['Low', 'Medium', 'High']
        else:
            buffer_range = abs(max_value - min_value) * 0.1
            vis_params = {
                'min': min_value - buffer_range,
                'max': max_value + buffer_range,
                'palette': ['blue', 'cyan', 'green', 'yellow', 'red']
            }
            legend_labels = None

        map_id = masked_mean.getMapId(vis_params)

        return jsonify({
            'tile_url': map_id['tile_fetcher'].url_format,
            'min': f"{min_value:.2e}",
            'max': f"{max_value:.2e}",
            'min_raw': min_value,
            'max_raw': max_value,
            'unit': unit,
            'legend_labels': legend_labels,
            'time_series': time_series
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
        
# CPCB AQI breakpoint table for different pollutants
###########################################################################################
CPCB_AQI_BREAKPOINTS = {
    'PM2.5': [
        {'bp_low': 0, 'bp_high': 30, 'aqi_low': 0, 'aqi_high': 50},
        {'bp_low': 31, 'bp_high': 60, 'aqi_low': 51, 'aqi_high': 100},
        {'bp_low': 61, 'bp_high': 90, 'aqi_low': 101, 'aqi_high': 200},
        {'bp_low': 91, 'bp_high': 120, 'aqi_low': 201, 'aqi_high': 300},
        {'bp_low': 121, 'bp_high': 250, 'aqi_low': 301, 'aqi_high': 400},
        {'bp_low': 251, 'bp_high': 1000, 'aqi_low': 401, 'aqi_high': 500}
    ],
    'PM10': [
        {'bp_low': 0, 'bp_high': 50, 'aqi_low': 0, 'aqi_high': 50},
        {'bp_low': 51, 'bp_high': 100, 'aqi_low': 51, 'aqi_high': 100},
        {'bp_low': 101, 'bp_high': 250, 'aqi_low': 101, 'aqi_high': 200},
        {'bp_low': 251, 'bp_high': 350, 'aqi_low': 201, 'aqi_high': 300},
        {'bp_low': 351, 'bp_high': 430, 'aqi_low': 301, 'aqi_high': 400},
        {'bp_low': 431, 'bp_high': 1000, 'aqi_low': 401, 'aqi_high': 500}
    ],
    'SO2': [
        {'bp_low': 0, 'bp_high': 40, 'aqi_low': 0, 'aqi_high': 50},
        {'bp_low': 41, 'bp_high': 80, 'aqi_low': 51, 'aqi_high': 100},
        {'bp_low': 81, 'bp_high': 380, 'aqi_low': 101, 'aqi_high': 200},
        {'bp_low': 381, 'bp_high': 800, 'aqi_low': 201, 'aqi_high': 300},
        {'bp_low': 801, 'bp_high': 1600, 'aqi_low': 301, 'aqi_high': 400},
        {'bp_low': 1601, 'bp_high': 2000, 'aqi_low': 401, 'aqi_high': 500}
    ],
    'NO2': [
        {'bp_low': 0, 'bp_high': 40, 'aqi_low': 0, 'aqi_high': 50},
        {'bp_low': 41, 'bp_high': 80, 'aqi_low': 51, 'aqi_high': 100},
        {'bp_low': 81, 'bp_high': 180, 'aqi_low': 101, 'aqi_high': 200},
        {'bp_low': 181, 'bp_high': 280, 'aqi_low': 201, 'aqi_high': 300},
        {'bp_low': 281, 'bp_high': 400, 'aqi_low': 301, 'aqi_high': 400},
        {'bp_low': 401, 'bp_high': 2000, 'aqi_low': 401, 'aqi_high': 500}
    ],
    'O3': [
        {'bp_low': 0, 'bp_high': 50, 'aqi_low': 0, 'aqi_high': 50},
        {'bp_low': 51, 'bp_high': 100, 'aqi_low': 51, 'aqi_high': 100},
        {'bp_low': 101, 'bp_high': 168, 'aqi_low': 101, 'aqi_high': 200},
        {'bp_low': 169, 'bp_high': 208, 'aqi_low': 201, 'aqi_high': 300},
        {'bp_low': 209, 'bp_high': 748, 'aqi_low': 301, 'aqi_high': 400},
        {'bp_low': 749, 'bp_high': 1000, 'aqi_low': 401, 'aqi_high': 500}
    ],
    'CO': [
        {'bp_low': 0, 'bp_high': 1, 'aqi_low': 0, 'aqi_high': 50},
        {'bp_low': 1.1, 'bp_high': 2, 'aqi_low': 51, 'aqi_high': 100},
        {'bp_low': 2.1, 'bp_high': 10, 'aqi_low': 101, 'aqi_high': 200},
        {'bp_low': 10.1, 'bp_high': 17, 'aqi_low': 201, 'aqi_high': 300},
        {'bp_low': 17.1, 'bp_high': 34, 'aqi_low': 301, 'aqi_high': 400},
        {'bp_low': 34.1, 'bp_high': 50, 'aqi_low': 401, 'aqi_high': 500}
    ]
}

def create_aqi_calculation(pollutant, image):
    """Create AQI calculation for an image using Earth Engine operations."""
    if pollutant == 'PM2.5':
        return ee.Image(0).expression(
            'concentration <= 30 ? ((50 - 0) / (30 - 0)) * (concentration - 0) + 0 : \
            concentration <= 60 ? ((100 - 51) / (60 - 31)) * (concentration - 31) + 51 : \
            concentration <= 90 ? ((200 - 101) / (90 - 61)) * (concentration - 61) + 101 : \
            concentration <= 120 ? ((300 - 201) / (120 - 91)) * (concentration - 91) + 201 : \
            concentration <= 250 ? ((400 - 301) / (250 - 121)) * (concentration - 121) + 301 : \
            ((500 - 401) / (1000 - 251)) * (concentration - 251) + 401',
            {'concentration': image}
        )
    elif pollutant == 'PM10':
        return ee.Image(0).expression(
            'concentration <= 50 ? ((50 - 0) / (50 - 0)) * (concentration - 0) + 0 : \
            concentration <= 100 ? ((100 - 51) / (100 - 51)) * (concentration - 51) + 51 : \
            concentration <= 250 ? ((200 - 101) / (250 - 101)) * (concentration - 101) + 101 : \
            concentration <= 350 ? ((300 - 201) / (350 - 251)) * (concentration - 251) + 201 : \
            concentration <= 430 ? ((400 - 301) / (430 - 351)) * (concentration - 351) + 301 : \
            ((500 - 401) / (1000 - 431)) * (concentration - 431) + 401',
            {'concentration': image}
        )
    elif pollutant == 'SO2':
        return ee.Image(0).expression(
            'concentration <= 40 ? ((50 - 0) / (40 - 0)) * (concentration - 0) + 0 : \
            concentration <= 80 ? ((100 - 51) / (80 - 41)) * (concentration - 41) + 51 : \
            concentration <= 380 ? ((200 - 101) / (380 - 81)) * (concentration - 81) + 101 : \
            concentration <= 800 ? ((300 - 201) / (800 - 381)) * (concentration - 381) + 201 : \
            concentration <= 1600 ? ((400 - 301) / (1600 - 801)) * (concentration - 801) + 301 : \
            ((500 - 401) / (2000 - 1601)) * (concentration - 1601) + 401',
            {'concentration': image}
        )
    elif pollutant == 'NO2':
        return ee.Image(0).expression(
            'concentration <= 40 ? ((50 - 0) / (40 - 0)) * (concentration - 0) + 0 : \
            concentration <= 80 ? ((100 - 51) / (80 - 41)) * (concentration - 41) + 51 : \
            concentration <= 180 ? ((200 - 101) / (180 - 81)) * (concentration - 81) + 101 : \
            concentration <= 280 ? ((300 - 201) / (280 - 181)) * (concentration - 181) + 201 : \
            concentration <= 400 ? ((400 - 301) / (400 - 281)) * (concentration - 281) + 301 : \
            ((500 - 401) / (2000 - 401)) * (concentration - 401) + 401',
            {'concentration': image}
        )
    elif pollutant == 'O3':
        return ee.Image(0).expression(
            'concentration <= 50 ? ((50 - 0) / (50 - 0)) * (concentration - 0) + 0 : \
            concentration <= 100 ? ((100 - 51) / (100 - 51)) * (concentration - 51) + 51 : \
            concentration <= 168 ? ((200 - 101) / (168 - 101)) * (concentration - 101) + 101 : \
            concentration <= 208 ? ((300 - 201) / (208 - 169)) * (concentration - 169) + 201 : \
            concentration <= 748 ? ((400 - 301) / (748 - 209)) * (concentration - 209) + 301 : \
            ((500 - 401) / (1000 - 749)) * (concentration - 749) + 401',
            {'concentration': image}
        )
    elif pollutant == 'CO':
        return ee.Image(0).expression(
            'concentration <= 1 ? ((50 - 0) / (1 - 0)) * (concentration - 0) + 0 : \
            concentration <= 2 ? ((100 - 51) / (2 - 1.1)) * (concentration - 1.1) + 51 : \
            concentration <= 10 ? ((200 - 101) / (10 - 2.1)) * (concentration - 2.1) + 101 : \
            concentration <= 17 ? ((300 - 201) / (17 - 10.1)) * (concentration - 10.1) + 201 : \
            concentration <= 34 ? ((400 - 301) / (34 - 17.1)) * (concentration - 17.1) + 301 : \
            ((500 - 401) / (50 - 34.1)) * (concentration - 34.1) + 401',
            {'concentration': image}
        )
    else:
        raise ValueError(f'Unsupported pollutant for AQI calculation: {pollutant}')


#Original
def calculate_aqi(value, pollutant):
    """Compute AQI based on PM2.5 using breakpoints."""
    if pollutant == 'PM2.5':
        if value <= 30:
            return ((50 - 0) / (30 - 0)) * (value - 0) + 0
        elif value <= 60:
            return ((100 - 51) / (60 - 31)) * (value - 31) + 51
        elif value <= 90:
            return ((200 - 101) / (90 - 61)) * (value - 61) + 101
        elif value <= 120:
            return ((300 - 201) / (120 - 91)) * (value - 91) + 201
        elif value <= 250:
            return ((400 - 301) / (250 - 121)) * (value - 121) + 301
        else:
            return ((500 - 401) / (500 - 251)) * (value - 251) + 401
    elif pollutant == 'PM10':
        if value <= 50:
            return ((50 - 0) / (50 - 0)) * (value - 0) + 0
        elif value <= 100:
            return ((100 - 51) / (100 - 51)) * (value - 51) + 51
        elif value <= 250:
            return ((200 - 101) / (250 - 101)) * (value - 101) + 101
        elif value <= 350:
            return ((300 - 201) / (350 - 251)) * (value - 251) + 201
        elif value <= 430:
            return ((400 - 301) / (430 - 351)) * (value - 351) + 301
        else:
            return ((500 - 401) / (1000 - 431)) * (value - 431) + 401
    elif pollutant == 'SO2':
        if value <= 40:
            return ((50 - 0) / (40 - 0)) * (value - 0) + 0
        elif value <= 80:
            return ((100 - 51) / (80 - 41)) * (value - 41) + 51
        elif value <= 380:
            return ((200 - 101) / (380 - 81)) * (value - 81) + 101
        elif value <= 800:
            return ((300 - 201) / (800 - 381)) * (value - 381) + 201
        elif value <= 1600:
            return ((400 - 301) / (1600 - 801)) * (value - 801) + 301
        else:
            return ((500 - 401) / (2000 - 1601)) * (value - 1601) + 401
    elif pollutant == 'NO2':
        if value <= 40:
            return ((50 - 0) / (40 - 0)) * (value - 0) + 0
        elif value <= 80:
            return ((100 - 51) / (80 - 41)) * (value - 41) + 51
        elif value <= 180:
            return ((200 - 101) / (180 - 81)) * (value - 81) + 101
        elif value <= 280:
            return ((300 - 201) / (280 - 181)) * (value - 181) + 201
        elif value <= 400:
            return ((400 - 301) / (400 - 281)) * (value - 281) + 301
        else:
            return ((500 - 401) / (2000 - 401)) * (value - 401) + 401
    elif pollutant == 'O3':
        if value <= 50:
            return ((50 - 0) / (50 - 0)) * (value - 0) + 0
        elif value <= 100:
            return ((100 - 51) / (100 - 51)) * (value - 51) + 51
        elif value <= 168:
            return ((200 - 101) / (168 - 101)) * (value - 101) + 101
        elif value <= 208:
            return ((300 - 201) / (208 - 169)) * (value - 169) + 201
        elif value <= 748:
            return ((400 - 301) / (748 - 209)) * (value - 209) + 301
        else:
            return ((500 - 401) / (2000 - 749)) * (value - 749) + 401
         
    elif pollutant == 'CO':
        if value <= 1:
            return ((50 - 0) / (1 - 0)) * (value - 0) + 0
        elif value <= 2:
            return ((100 - 51) / (2 - 1.1)) * (value - 1.1) + 51
        elif value <= 10:
            return ((200 - 101) / (10 - 2.1)) * (value - 2.1) + 101
        elif value <= 17:
            return ((300 - 201) / (17 - 10.1)) * (value - 10.1) + 201
        elif value <= 34:
            return ((400 - 301) / (34 - 17.1)) * (value - 17.1) + 301
        else:
            return ((500 - 401) / (50 - 34.1)) * (value - 34.1) + 401    
    else:
        raise ValueError(f'Unsupported pollutant for AQI calculation: {pollutant}')
    
#Compute AQI by date
def compute_aqi_by_date(stats):
    """Compute AQI for each date separately."""
    # Step 1: Flatten the list
    flattened_data = [entry for sublist in stats for entry in sublist]
    print(flattened_data)

    # Step 2: Group data by date and pollutant
    datewise_pollutant_values = defaultdict(lambda: defaultdict(list))
    print(datewise_pollutant_values)

    for entry in flattened_data:
        date = entry["date"]
        pollutant = entry["pollutant"]
        datewise_pollutant_values[date][pollutant].append(entry["mean_value"])
    # Step 3: Process values (averaging if needed)
    for date, pollutants in datewise_pollutant_values.items():
        for pollutant, values in pollutants.items():
            if len(values) > 1:
                datewise_pollutant_values[date][pollutant] = sum(values) / len(values)  # Compute average
            else:
                datewise_pollutant_values[date][pollutant] = values[0]  # Keep single value

        

    # Step 3: Compute AQI per date
    datewise_aqi = {}

    for date, pollutants in datewise_pollutant_values.items():
        pollutant_aqi = {}
        for pollutant, values in pollutants.items():
            p95_value = np.percentile(values, 95)  # Compute p95 for each pollutant
            aqi = create_aqi_calculation(p95_value, pollutant)
            pollutant_aqi[pollutant] = aqi
        
        datewise_aqi[date] = pollutant_aqi
    return datewise_aqi

def compute_avg_aqi_by_date(datewise_aqi):
    avg_aqi_by_date = {}

    for date, pollutants in datewise_aqi.items():
        # Check if PM2.5 or PM10 exists and is not None
        pm25 = pollutants.get('PM2.5')
        pm10 = pollutants.get('PM10')

        if pm25 is not None or pm10 is not None:
            # Filter valid pollutant values (non-None)
            valid_values = [v for v in pollutants.values() if v is not None]
            
            if len(valid_values) >= 3:
                avg_aqi_by_date[date] = max(valid_values)
            else:
                print(f"{date}: Insufficient Data (less than 3 pollutants)")
        else:
            print(f"{date}: No PM2.5 or PM10 data")
    
    return avg_aqi_by_date

#AQI categorizer
def categorize_aqi(aqi):
    """Return AQI category based on value."""
    if aqi <= 50:
        return 'Good'
    elif aqi <= 100:
        return 'Moderate'
    elif aqi <= 150:
        return 'Unhealthy for Sensitive Groups'
    elif aqi <= 200:
        return 'Unhealthy'
    elif aqi <= 300:
        return 'Very Unhealthy'
    else:
        return 'Hazardous'


# Function for Processing the display of AQI for each of the Pollutants within the specific bounds or geometry
##########################################################################################

def process_pollutant_aqi(geometry_data, pollutant, start_date, end_date, scale=1000):
    """Process pollutant data and calculate AQI for a given geometry."""
    if pollutant not in POLLUTANT_CONFIGS:
        raise ValueError(f'Unsupported pollutant for AQI calculation: {pollutant}')

    config = POLLUTANT_CONFIGS[pollutant]
    bounds = geometry_data['bounds']
    geometry = geometry_data['geometry']

    # Get the concentration data
    collection = ee.ImageCollection(config['collection']) \
        .filterBounds(bounds) \
        .filterDate(start_date, end_date) \
        .select(config['band'])
    

    # Mask negative values
    def mask_negative_values(image):
        return image.updateMask(image.gte(0))
    
    collection = collection.map(mask_negative_values)
    # Calculate mean concentration
    mean_image = collection.mean()
    
    # Apply scale factor and offset
    if 'scale_factor' in config:
        mean_image = mean_image.multiply(config['scale_factor'])
        if 'offset' in config:
            mean_image = mean_image.add(config['offset'])

    # Special handling for CO to convert from mol/m² to mg/m³ (approximate conversion)
    """if pollutant == 'CO':
        # Convert CO from mol/m² to mg/m³
        # Molecular weight of CO = 28.01 g/mol
        # Standard atmospheric conditions
        mean_image = mean_image.multiply(28.01).divide(24.45)"""

    # Calculate AQI for the image
    aqi_image = create_aqi_calculation(pollutant, mean_image)
    
    # Clip to geometry and mask values outside 0-500 range
    masked_aqi = aqi_image.clip(geometry) \
        .updateMask(aqi_image.gte(0).And(aqi_image.lte(500)))
    

    # Handle PM2.5 band naming
    if pollutant == 'PM2.5':
        masked_aqi = masked_aqi.rename('PM2_5_AQI')
    else:
        masked_aqi = masked_aqi.rename(f'{pollutant}_AQI')

    # Calculate AQI statistics
    stats = masked_aqi.reduceRegion(
        reducer=ee.Reducer.percentile([5, 95]),
        geometry=geometry,
        scale=scale,
        maxPixels=1e8,
        bestEffort=True
    ).getInfo()

    # Handle PM2.5 statistics key
    if pollutant == 'PM2.5':
        stats = {
            'PM2_5_AQI_p5': stats.get('PM2_5_AQI_p5'),
            'PM2_5_AQI_p95': stats.get('PM2_5_AQI_p95')
        }
    print(pollutant)
    print(stats)
    #print(dates) 
    return masked_aqi, stats, 'AQI' #,dates


# API Route for getting the AQI of the City
########################################################################################

@app.route('/api/get-aqi-pollutant-city', methods=['GET'])
 
def get_pollutant_city_aqi():
    try:
        # Extract parameters
        city = request.args.get('city')
        print(city)
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        pollutant = request.args.get('pollutant')

        if not all([city, start_date, end_date, pollutant]):
            return jsonify({'error': 'Missing required parameters'}), 400

        # Get optimized city geometry
        city_data = get_optimized_geometry(
            f"static/city/{city.upper()}.geojson",
            simplify_error=100
        )
        
        #Process AQI data
        masked_aqi, stats, unit = process_pollutant_aqi(
            city_data, pollutant, start_date, end_date, scale=1000
        )
        time_series = compute_timeseries_mean_2(city_data, pollutant, start_date, end_date)
        if time_series is None:
            return 0
        else:
            for entry in time_series:
                value = entry.get('value')
                print("Concentration",value)
                if value is not None:
                    entry['aqi'] = calculate_aqi(value, pollutant)
            # Extract min/max AQI values, handling PM2.5 special case
            if pollutant == 'PM2.5':
                min_value = stats.get('PM2_5_AQI_p5', None)
                max_value = stats.get('PM2_5_AQI_p95', None)
            else:
                min_value = stats.get(f'{pollutant}_AQI_p5', None)
                max_value = stats.get(f'{pollutant}_AQI_p95', None)

            if min_value is None or max_value is None:
                return jsonify({'error': f'Could not calculate AQI range for {pollutant}.'}), 500

            # Set visualization parameters with AQI color scheme
            vis_params = {
                'min': 0.001,
                'max': 500,
                'palette': ['#00ff00', '#ffff00', '#ff9933', '#ff0000', '#990066', '#990000']
            }
            
            legend_labels = ['Good (0-50)', 'Satisfactory (51-100)', 'Moderate (101-200)', 
                            'Poor (201-300)', 'Very Poor (301-400)', 'Severe (401-500)']

            # Generate map
            map_id = masked_aqi.getMapId(vis_params)
            print(time_series)
            return jsonify({
                'tile_url': map_id['tile_fetcher'].url_format,
                'min': f"{min_value:.2f}",
                'max': f"{max_value:.2f}",
                'min_raw': min_value,
                'max_raw': max_value,
                'unit': unit,
                'legend_labels': legend_labels,
                'pollutant': pollutant,  # Adding pollutant to the response
                'timeseries' : time_series
            })

    except Exception as e:
        print(f"Error in get_pollutant_city_aqi: {traceback.format_exc()}")  # Add debug logging
        return jsonify({'error': str(e)}), 500
 
# API Route for getting the AQI of the State
########################################################################################
    
@app.route('/api/get-aqi-pollutant-state', methods=['GET'])
def get_pollutant_state_aqi():
    try:
        # Extract parameters
        state = request.args.get('state')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        pollutant = request.args.get('pollutant')
        if not all([state, start_date, end_date, pollutant]):
            return jsonify({'error': 'Missing required parameters'}), 400

        # Get optimized city geometry
        state_data = get_optimized_geometry(
            f"static/state/{state}.geojson",
            simplify_error=100
        )
        
        # Process AQI data
        masked_aqi, stats, unit = process_pollutant_aqi(
            state_data, pollutant, start_date, end_date, scale=1000
        )
        time_series = compute_timeseries_mean_2(state_data, pollutant, start_date, end_date)
        print(time_series)
        for entry in time_series:
            value = entry.get('value')
            if value is not None:
                entry['aqi'] = calculate_aqi(value, pollutant)
        # Extract min/max AQI values, handling PM2.5 special case
        if pollutant == 'PM2.5':
            min_value = stats.get('PM2_5_AQI_p5', None)
            max_value = stats.get('PM2_5_AQI_p95', None)
        else:
            min_value = stats.get(f'{pollutant}_AQI_p5', None)
            max_value = stats.get(f'{pollutant}_AQI_p95', None)

        if min_value is None or max_value is None:
            return jsonify({'error': f'Could not calculate AQI range for {pollutant}.'}), 500
        #print(time_series)
        # Set visualization parameters with AQI color scheme
        vis_params = {
            'min': 0,
            'max': 500,
            'palette': ['#00ff00', '#ffff00', '#ff9933', '#ff0000', '#990066', '#990000']
        }
        
        legend_labels = ['Good (0-50)', 'Satisfactory (51-100)', 'Moderate (101-200)', 
                        'Poor (201-300)', 'Very Poor (301-400)', 'Severe (401-500)']

        # Generate map
        map_id = masked_aqi.getMapId(vis_params)

        return jsonify({
            'tile_url': map_id['tile_fetcher'].url_format,
            'min': f"{min_value:.2f}",
            'max': f"{max_value:.2f}",
            'min_raw': min_value,
            'max_raw': max_value,
            'unit': unit,
            'legend_labels': legend_labels,
            'pollutant': pollutant,
            'timeseries': time_series
        })

    except Exception as e:
        print(f"Error in get_pollutant_state_aqi: {str(e)}")  #debug logging
        return jsonify({'error': str(e)}), 500
    

# API Route for Processing the pixel based time series analysis of the pollutants
#######################################################################################
@app.route('/api/get-time-series', methods=['GET'])
def get_time_series():
    try:
        # Extract and validate input parameters
        lat = request.args.get('lat', type=float)
        lon = request.args.get('lon', type=float)
        pollutant = request.args.get('pollutant')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        if not all([lat, lon, pollutant, start_date, end_date]):
            return jsonify({'error': 'Missing required parameters.'}), 400

        if pollutant not in POLLUTANT_CONFIGS:
            return jsonify({'error': f'Invalid pollutant: {pollutant}'}), 400

        # Create a point geometry with a small buffer to ensure data capture
        point = ee.Geometry.Point([lon, lat]).buffer(1000)

        # Get the pollutant configuration
        config = POLLUTANT_CONFIGS[pollutant]

        # Create the base collection
        collection = ee.ImageCollection(config['collection'])\
            .filterDate(start_date, end_date)\
            .filterBounds(point)\
            .select(config['band'])

        # Add negative value masking
        def mask_negative_values(image):
            return image.updateMask(image.gte(0))
        
        collection = collection.map(mask_negative_values)

        # Debug: Log the collection size
        collection_size = collection.size().getInfo()
        print(f"Collection size: {collection_size}")

        if collection_size == 0:
            return jsonify({'error': 'No data available for the specified dates.'}), 404

        # Get all unique dates in the collection
        def get_image_date(image):
            date = ee.Date(image.get('system:time_start'))
            return ee.Feature(None, {'date': date.format('YYYY-MM-dd')})

        dates = collection.map(get_image_date).distinct('date').aggregate_array('date').getInfo()
        
        # Process each image in the collection
        def process_image(image):
            date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')
            
            # Determine the correct band name
            band_name = config['band']  # Use the default band from configuration
            
            # Safely get the value using the selected band
            value = image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=1000,
                maxPixels=1e9
            ).get(band_name)
            
            return ee.Feature(None, {
                'date': date,
                'value': value
            })

        features = collection.map(process_image)
        time_series_data = features.getInfo().get('features', [])

        # Process the data and apply scaling factors
        series_data = []
        for feature in time_series_data:
            props = feature['properties']
            value = props.get('value')
            
            if value is not None and not isinstance(value, str):  # Ensure value is numeric and not None
                if value >= 0:  # Additional check for negative values
                    if 'scale_factor' in config:
                        value = value * config['scale_factor'] + config.get('offset', 0)
                    
                    # Round the value to 4 decimal places for cleaner data
                    value = round(float(value), 4)
                    
                    series_data.append({
                        'date': props['date'],
                        'value': value
                    })

        # Sort the data by date
        series_data.sort(key=lambda x: x['date'])

        # Remove duplicates while keeping the first occurrence
        seen_dates = set()
        unique_series_data = []
        for data_point in series_data:
            if data_point['date'] not in seen_dates:
                seen_dates.add(data_point['date'])
                unique_series_data.append(data_point)

        return jsonify({
            'series': unique_series_data,
            'unit': config.get('unit', 'unknown'),
            'dates': dates
        })

    except ee.EEException as gee_error:
        print(f"GEE Error: {str(gee_error)}")
        return jsonify({'error': f'Google Earth Engine error: {str(gee_error)}'}), 500
    except Exception as e:
        print(f"General Error: {str(e)}")
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500
    
def indv_chart(plt_charts):
    plt_aqi_chart = defaultdict(list) 
    for date, pollutants in plt_charts.items():
        for pollutant, value in pollutants.items():
            plt_aqi_chart[pollutant].append((date, value))
    return dict(plt_aqi_chart)   
# # API Route for Showing the Final AQI for the City or the State
# ########################################################################################
@app.route('/api/show-final-aqi', methods=['GET'])
def show_final_aqi():
    try:
        # Extract parameters
        city = request.args.get('city')
        state = request.args.get('state')
        pollutants = request.args.get('pollutant')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        if not (city or state):
            return jsonify({'error': 'Either city or state must be provided'}), 400

        pollutant_list = pollutants.split(',')
        if len(pollutant_list) < 3:
            return jsonify({'error': 'Please select at least three valid pollutants'}), 400

        # Load and simplify geometry
        geojson_path = f"static/{'city/' + city.upper() if city else 'state/' + state}.geojson"
        if not os.path.exists(geojson_path):
            return jsonify({'error': f"GeoJSON file for {city or state} not found."}), 404

        geo_data = get_optimized_geometry(geojson_path, simplify_error=200)
        region = geo_data['geometry']

        # Visualization setup (palette + legend only)
        palette = ['#00ff00', '#ffff00', '#ff9933', '#ff0000', '#990066', '#990000']
        legend_labels = [
            'Good (0-50)', 'Satisfactory (51-100)', 'Moderate (101-200)',
            'Poor (201-300)', 'Very Poor (301-400)', 'Severe (401-500)'
        ]

        image_list = []
        Aqi_chart = {}
        # Compute time series for PM2.5 and PM10 separately
       # Precompute PM2.5 and PM10 time series if they are in the list
        pm25_series = None
        pm10_series = None

        if 'PM2.5' in pollutant_list:
            pm25_series = compute_timeseries_mean_2(geo_data, 'PM2.5', start_date, end_date)

        if 'PM10' in pollutant_list:
            pm10_series = compute_timeseries_mean_2(geo_data, 'PM10', start_date, end_date)

        # If both are present but both are None → error
        if 'PM2.5' in pollutant_list and 'PM10' in pollutant_list:
            if pm25_series is None and pm10_series is None:
                raise ValueError("Both PM2.5 and PM10 returned no data. Cannot proceed.")

        # Start processing all pollutants
        for pollutant in pollutant_list:
            print(f"Processing pollutant: {pollutant}")

            # Use precomputed values for PM2.5 and PM10
            if pollutant == 'PM2.5':
                if pm25_series is None:
                    print("PM2.5 data is missing. Skipping.")
                    continue
                time_series = pm25_series

            elif pollutant == 'PM10':
                if pm10_series is None:
                    print("PM10 data is missing. Skipping.")
                    continue
                time_series = pm10_series

            else:
                time_series = compute_timeseries_mean_2(geo_data, pollutant, start_date, end_date)
                if time_series is None:
                    print(f"No data for {pollutant}. Skipping.")
                    continue

            # Process AQI and time series
            masked_aqi, stats, _ = process_pollutant_aqi(geo_data, pollutant, start_date, end_date, scale=1000)
            print(stats)

            for entry in time_series:
                value = entry.get('value')
                date = entry.get('date')
                if value is not None and date:
                    aqi = calculate_aqi(value, pollutant)
                    if date not in Aqi_chart:
                        Aqi_chart[date] = {}
                    Aqi_chart[date][pollutant] = aqi

            # Standardize and collect AQI image
            standardized_image = (
                masked_aqi.select([0], ['AQI'])
                .unmask(0)
                .toFloat()
            )
            image_list.append(standardized_image)
        indv_plts = indv_chart(Aqi_chart)
        avg_aqi_by_date = compute_avg_aqi_by_date(Aqi_chart)
        final_aqi_ic = ee.ImageCollection.fromImages(image_list)
        final_mean = final_aqi_ic.max()

        # Compute p5-p95 range for visualization
        reduced = final_mean.reduceRegion(
            reducer=ee.Reducer.percentile([5, 95]),
            geometry=region,
            scale=1000,
            maxPixels=1e13
        )

        min_val = reduced.get('AQI_p5').getInfo()
        max_val = reduced.get('AQI_p95').getInfo()

        # Safe fallback if reduction failed
        if min_val is None or max_val is None:
            min_val, max_val = 0, 500

        vis_params = {
            'min': min_val,
            'max': max_val,
            'palette': palette
        }

        final_map_id = final_mean.getMapId(vis_params)

        return jsonify({
            'tile_url': final_map_id['tile_fetcher'].url_format,
            'mapid': final_map_id['mapid'],
            'token': final_map_id['token'],
            'min': f"{min_val:.2f}",
            'max': f"{max_val:.2f}",
            'min_raw': min_val,
            'max_raw': max_val,
            'legend_labels': legend_labels,
            'timeseries':time_series,
            'Final_chart':avg_aqi_by_date,
            'indv_pult':indv_plts
        })
    #
    #'Aqichart':Aqi_chart,
    #'unit': 'AQI',

    except Exception as e:
        return jsonify({'error': str(e)}), 500


##########   Tab-3   ##############################
def get_optimized_geometry_2(geojson_path):
    geojson_data = gpd.read_file(geojson_path)
    geometry = geojson_data.union_all()
    return {
        'geometry': mapping(geometry)
    }


@app.route('/api/get-aqi-health-state', methods=['GET'])
def aqi_health():
    state = request.args.get('state')
    year = request.args.get('year')
    cause = request.args.get('cause')

    print(state, year, cause)

    if not state or not year or not cause:
        return jsonify({'error': 'State, year and cause must be provided'}), 400

    pth = rf"static/state/{state}.geojson"
    geo_data = get_optimized_geometry_2(pth)

    df = pd.read_csv("static/2021_filtered.csv")
    df['year'] = pd.to_numeric(df['year'], errors='coerce')

    year = int(year)

    filtered = df[
        (df['location_name'] == state) &
        (df['year'] == year) &
        (df['cause_name'] == cause)
    ]

    if filtered.empty:
        return jsonify({'error': f'No data for {state} in {year} and {cause}'}), 404

    # ✅ Correct groupby
    series2 = (
    filtered[
        (filtered['metric_name'] == 'Rate') &
        (filtered['cause_name'] == cause)&
        (filtered['measure_name'] == 'Deaths')
    ]
    .groupby('age_name')['val']
    .sum()
    )

    print(series2)

    series = (
    filtered[
        (filtered['metric_name'] == 'Rate') &
        (filtered['cause_name'] == cause) &
        (filtered['age_name'] == 'All ages')&
        (filtered['measure_name'] == 'Deaths')
        ]['val']
        .sum()
    )

    print(series)

    data = float(series)

    return jsonify({
        'state': state,
        'year': year,
        'cause': cause,
        'value': data,
        'geometry': geo_data['geometry'],
        'series': series2.to_dict()   # ✅ JSON-safe
    })



# API route to get the Windy API key
@app.route('/api/get-windy-api-key', methods=['GET'])
def get_windy_api_key():
    if WINDY_API_KEY:
        return jsonify({'api_key': WINDY_API_KEY})
    else:
        return jsonify({'error': 'Windy API key not configured.'}), 500


# Chatbot using the Natural Language Toolkit (nltk) module package
####################################################################

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class AirQualityChatbot:
    def __init__(self, app):
        self.app = app
        try:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            self.lemmatizer = lambda x: x  # Simple pass-through function
            self.stop_words = set()
            print("Warning: NLTK resources not fully available.")
        
        self.pollutant_info = {
            'SO2': 'Sulfur dioxide (SO2) is a toxic gas with a pungent odor. It\'s primarily produced from the burning of fossil fuels containing sulfur.',
            'NO2': 'Nitrogen dioxide (NO2) is a reddish-brown gas that primarily comes from the burning of fuel.',
            'CO': 'Carbon monoxide (CO) is a colorless, odorless gas that\'s produced by incomplete combustion of carbon-based fuels.',
            'O3': 'Ozone (O3) at ground level is a harmful air pollutant and a key component of smog.',
            'PM2.5': 'PM2.5 refers to fine particulate matter smaller than 2.5 micrometers in diameter.',
            'PM10': 'PM10 refers to particulate matter up to 10 micrometers in size.',
            'HCHO': 'Formaldehyde (HCHO) is a colorless gas that can cause irritation to the eyes, nose, and throat.'
        }

        self.city_coordinates = {
            'hyderabad': {'lat': 17.3850, 'lon': 78.4867},
            'srinagar': {'lat': 34.0837, 'lon': 74.7973},
            'delhi': {'lat': 28.6139, 'lon': 77.2090},
            'mumbai': {'lat': 19.0760, 'lon': 72.8777},
            'bangalore': {'lat': 12.9716, 'lon': 77.5946}
        }

    def preprocess_text(self, text):
        try:
            tokens = word_tokenize(text.lower())
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
            return tokens
        except Exception as e:
            print(f"Error in text preprocessing: {str(e)}")
            return text.lower().split()

    def extract_date_range(self, text):
        date_patterns = [
            r'(\d{1,2}(?:st|nd|rd|th)?\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{4})',
            r'(\d{2}/\d{2}/\d{4})',  # e.g., 01/01/2024
            r'(\d{4}-\d{2}-\d{2})'   # e.g., 2024-01-01
        ]
        
        dates = []
        for pattern in date_patterns:
            found_dates = re.findall(pattern, text.lower())
            dates.extend(found_dates)
        
        if len(dates) >= 2:
            try:
                if "/" in dates[0]:
                    start_date = datetime.strptime(dates[0], '%d/%m/%Y').strftime('%Y-%m-%d')
                else:
                    start_date = datetime.strptime(dates[0], '%d %B %Y').strftime('%Y-%m-%d')
                
                if "/" in dates[1]:
                    end_date = datetime.strptime(dates[1], '%d/%m/%Y').strftime('%Y-%m-%d')
                else:
                    end_date = datetime.strptime(dates[1], '%d %B %Y').strftime('%Y-%m-%d')

                return start_date, end_date
            except Exception as e:
                print(f"Error in date conversion: {e}")
        return None, None

    def extract_city(self, text):
        tokens = self.preprocess_text(text)
        for city in self.city_coordinates.keys():
            if city in tokens:
                return city
        return None

    def extract_pollutant(self, text):
        tokens = self.preprocess_text(text)
        for pollutant in self.pollutant_info.keys():
            if pollutant.lower() in tokens:
                return pollutant
        return None

    def get_pollutant_data(self, city, pollutant, start_date, end_date):
        """Fetch pollutant data from an API endpoint."""
        with self.app.test_client() as client:
            city_coords = self.city_coordinates[city]
            response = client.get(f'/api/get-pollutant?lat={city_coords["lat"]}&lon={city_coords["lon"]}&pollutant={pollutant}&start_date={start_date}&end_date={end_date}')
            return response.get_json()

    def get_pollutant_stats(self, city, pollutant, start_date, end_date):
        """Get statistical information about pollutant levels."""
        data = self.get_pollutant_data(city, pollutant, start_date, end_date)
        if 'error' in data:
            return f"Sorry, I couldn't retrieve the data: {data['error']}"
        
        min_val = float(data['min'])
        max_val = float(data['max'])
        unit = data['unit']
        
        return {
            'min': min_val,
            'max': max_val,
            'unit': unit,
            'average': (min_val + max_val) / 2
        }

    def generate_response(self, user_input):
        city = self.extract_city(user_input)
        pollutant = self.extract_pollutant(user_input)
        start_date, end_date = self.extract_date_range(user_input)

        if 'what is' in user_input.lower() and pollutant and not city:
            return self.pollutant_info.get(pollutant, "I don't have information about that pollutant.")

        if city and pollutant and start_date and end_date:
            try:
                stats = self.get_pollutant_stats(city, pollutant, start_date, end_date)
                response = f"For {city.title()} between {start_date} and {end_date}:\n"
                response += f"The {pollutant} levels ranged from {stats['min']:.2f} to {stats['max']:.2f} {stats['unit']}\n"
                response += f"The average concentration was approximately {stats['average']:.2f} {stats['unit']}"
                return response
            except Exception as e:
                return f"I apologize, but I encountered an error while retrieving the data: {str(e)}"

        missing_info = []
        if not city:
            missing_info.append("city")
        if not pollutant:
            missing_info.append("pollutant type")
        if not start_date or not end_date:
            missing_info.append("date range")
        
        if missing_info:
            return f"I need more information to answer your question. Please specify the {', '.join(missing_info)}."

        return "I'm not sure how to help with that query. Please try asking about specific pollutant levels in a city for a particular date range."

# API Route for having a chat with the Chatbot
##############################################################
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        chatbot = AirQualityChatbot(app)
        response = chatbot.generate_response(user_message)
        
        return jsonify({'response': response})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

if __name__ == '__main__':
    app.run(debug=True)
