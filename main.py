import json
import breakup
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from trajectory import TrajectorySimulation


def main(config_path):
    with open(config_path) as f:
        config = json.load(f)
    lcPowerLawExponent = -2.6
    n = breakup.calculate_fragment_count(config["simulation"]["minimalCharacteristicLength"])
    c_l= breakup.calculateCharacteristicLengthFromMass(config["simulation"]["mass"])
    #a_m= breakup.calculate_area_mass_ratio(c_l, "ROCKET_BODY")
    area = breakup.calculateCircleArea(c_l)
    a_m= area / config["simulation"]["mass"]
    df= breakup.generate_fragements(n)
    df['Characteristic Length']= df['Characteristic Length'].apply(lambda x: breakup.calculate_characteristic_length(config["simulation"]["minimalCharacteristicLength"],
                                                                               c_l, lcPowerLawExponent))
    df['Satellite Type']= "DEBRIS"
    df['Name']= range(len(df))
    df["A/M"]= df["Characteristic Length"].apply(breakup.calculate_area_mass_ratio)
    df["area"]= df["Characteristic Length"].apply(breakup.calculate_area)
    df["mass"]= df["area"]/df["A/M"]
    pos_x,pos_y,pos_z= list(config["simulation"]["position"])
    vel_x,vel_y,vel_z= list(config["simulation"]["velocity"])
    df["pos_x"], df["pos_y"],df["pos_z"] =pos_x,pos_y,pos_z
    df["vel_x"], df["vel_y"],df["vel_z"] =vel_x,vel_y,vel_z
    df["eject_vel"]= df.apply(breakup.delta_velocity_distribution, axis=1)
    df[['eject_x','eject_y',"eject_z" ]] = pd.DataFrame(df["eject_vel"].tolist(), index= df.index)
    df["vel_x"]+=df["eject_x"]
    df["vel_y"]+=df["eject_y"]
    df["vel_z"]+=df["eject_z"]
    df= df.drop(['eject_vel'], axis=1)
    print(df)
    print(df["mass"].sum())

    df.plot.scatter(x="Characteristic Length",y="A/M")
    plt.yscale("log")
    plt.xscale("log")
    plt.savefig("result/len_to_am.png")

    plt.close()
    df["Characteristic Length"].plot.hist(bins=40)
    plt.xscale("log")
    plt.savefig("result/len.png")

    # Plotting the velocity vectors
    plt.close()

    # Adjusting the axis scaling based on the highest absolute value in the velocities
    max_abs_value = max(df['vel_x'].abs().max(), df['vel_y'].abs().max())
    limit = np.ceil(max_abs_value)
    plt.figure(figsize=(8, 8))
    plt.quiver(np.zeros(len(df)), np.zeros(len(df)), df['vel_x'], df['vel_y'], angles='xy', scale_units='xy', scale=1, color='r')
    plt.xlim(-limit, limit)
    plt.ylim(-limit, limit)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True)
    plt.xlabel('X Velocity')
    plt.ylabel('Y Velocity')
    plt.title('Velocity Vectors')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("speeds.png")
    plt.close()

    df.to_excel("result/df.xlsx")

    # Initialize and run the trajectory simulation
    simulation = TrajectorySimulation(df)
    ani = simulation.run()

    # Save the animation
    ani.save('result/debris_trajectories2.mp4', writer='ffmpeg', fps=100)

    
if __name__=="__main__":
    main("config.json")