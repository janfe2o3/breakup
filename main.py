import json
import breakup
import pandas as pd
import matplotlib.pyplot as plt

def main(config_path):
    with open(config_path) as f:
        config = json.load(f)
    lcPowerLawExponent = -2.6
    deltaVelocityFactorOffset = (0.2, 1.85)
    n = breakup.calculate_fragment_count(config["simulation"]["minimalCharacteristicLength"])
    c_l= breakup.calculateCharacteristicLengthFromMass(config["simulation"]["mass"])
    #a_m= breakup.calculate_area_mass_ratio(c_l, "ROCKET_BODY")
    area = breakup.calculateCircleArea(c_l)
    a_m= area / config["simulation"]["mass"]
    df= breakup.generate_fragements(n)
    df['Characteristic Length']= df['Characteristic Length'].apply(lambda x: breakup.calculate_characteristic_length(config["simulation"]["minimalCharacteristicLength"],
                                                                               c_l, lcPowerLawExponent))
    df['Satellite Type']= "DEBRIS"
    df['Name']= "DEBRIS"
    df["A/M"]= df["Characteristic Length"].apply(breakup.calculate_area_mass_ratio)
    df["area"]= df["Characteristic Length"].apply(breakup.calculate_area)
    df["mass"]= df["area"]/df["A/M"]
    pos_x,pos_y,pos_z= list(config["simulation"]["position"])
    vel_x,vel_y,vel_z= list(config["simulation"]["velocity"])
    df["pos_x"], df["pos_y"],df["pos_z"] =pos_x,pos_y,pos_z
    df["vel_x"], df["vel_y"],df["vel_z"] =vel_x,vel_y,vel_z
    print(df)
    print(df["mass"].sum())

    df.plot.scatter(x="Characteristic Length",y="A/M")
    plt.yscale("log")
    plt.xscale("log")
    plt.savefig("test.png")

    plt.close()
    df["Characteristic Length"].plot.hist(bins=40)
    plt.xscale("log")
    plt.savefig("len.png")
    print(df["Characteristic Length"].mean())
    print(df["A/M"].mean())





if __name__=="__main__":
    main("config.json")