import pandas as pd
import numpy as np
from utils import mu_1, sigma_1, mu_2, sigma_2, mu_soc, sigma_soc, alpha

import random

def calculate_fragment_count(L_c):
    n= int(6*L_c**(-1.6))
    return n

def generate_fragements(n):
    # Create an empty DataFrame with columns A, B, C, etc.
    columns_list = ["Name","Satellite Type", "Characteristic Length","A/M","area","mass","ejection Velocity","Velocity","pos_x", "pos_y", "pos_z"]
    df = pd.DataFrame(columns=columns_list, index=range(1, n + 1))

    lcPowerLawExponent = -2.6
    deltaVelocityFactorOffset = (0.2, 1.85)
    return df

def calculateCharacteristicLengthFromMass(mass):
    Mul92_937PI = 92.937 * np.pi
    Inv2_26 = 1.0 / 2.26
    return ((6.0 * mass) / (Mul92_937PI))**Inv2_26



def calculate_area_mass_ratio(characteristic_length, sat_type="ROCKET_BODY"):
    log_lc = np.log10(characteristic_length)
    print(sat_type)

    if characteristic_length > 0.11:
            # Case bigger than 11 cm
            n1 = np.random.normal(mu_1(sat_type, log_lc), sigma_1(sat_type, log_lc))
            print(mu_2(sat_type, log_lc),sigma_2(sat_type, log_lc))
            n2 = np.random.normal(mu_2(sat_type, log_lc), sigma_2(sat_type, log_lc))
            print(n2,"\n")
            return np.power(10.0, alpha(sat_type, log_lc) * n1 +
                            (1 - alpha(sat_type, log_lc)) * n2)
    elif characteristic_length < 0.08:
            # Case smaller than 8 cm
            n = np.random.normal(mu_soc(log_lc), sigma_soc(log_lc))
            return np.power(10.0, n)
    else:
            # Case between 8 cm and 11 cm
            n1 = np.random.normal(mu_1(sat_type, log_lc), sigma_1(sat_type, log_lc))
            n2 = np.random.normal(mu_2(sat_type, log_lc), sigma_2(sat_type, log_lc))
            n = np.random.normal(mu_soc(log_lc), sigma_soc(log_lc))
            
            y1 = np.power(10.0, alpha(sat_type, log_lc) * n1 +
                          (1.0 - alpha(sat_type, log_lc)) * n2)
            y0 = np.power(10.0, n)
            
    return y0 + (characteristic_length - 0.08) * (y1 - y0) / 0.03

def calculateCircleArea(characteristicLength):
    radius = characteristicLength / 2.0
    return np.pi * radius**2

def transform_uniform_to_power_law(x0, x1, n, y):
    step = (x1**(n + 1.0) - x0**(n + 1.0)) * y + x0**(n + 1.0)
    return step**(1.0 / (n + 1.0))

def calculate_characteristic_length(minimal_characteristic_length, maximal_characteristic_length, lc_power_law_exponent):
    y = np.random.uniform(0.0, 1.0)
    return transform_uniform_to_power_law(minimal_characteristic_length, maximal_characteristic_length, lc_power_law_exponent, y)

def calculateMass(area, areaToMassRatio):
    return area/areaToMassRatio

def calculate_area(characteristic_length):
    lc_bound = 0.00167
    if characteristic_length < lc_bound:
        factor_little = 0.540424
        return factor_little * characteristic_length ** 2
    else:
        exponent_big = 2.0047077
        factor_big = 0.556945
        return factor_big * characteristic_length ** exponent_big

import numpy as np
from numpy import log10
from scipy.stats import norm

# Assuming the necessary class attributes and methods are defined elsewhere in the class

def delta_velocity_distribution(vel_x, vel_y, vel_z,a_m):
    for vel in [vel_x, vel_y, vel_z]:
        # Calculate velocity as a scalar based on Equation 11/12
        chi = np.log10(a_m)
        mu = delta_velocity_factor_offset[0] * chi + delta_velocity_factor_offset[1]
        sigma = 0.4
        velocity_scalar = 10 ** np.random.normal(mu, sigma)

        ejection_velocity_vector = calculate_velocity_vector(velocity_scalar)

        # Update the velocity and ejection_velocity in the tuple
        # Since tuples are immutable, you might actually be working with a list or need to create a new tuple
        new_velocity = velocity + ejection_velocity_vector
        # Depending on how you need to update this, you might reassign it back to your data structure
        
        # Example of updating the tuple, assuming you're actually using a mutable structure like a list
        tuple[1], tuple[2] = new_velocity, ejection_velocity_vector

