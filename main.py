import json
import breakup
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

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
    plt.savefig("test.png")

    plt.close()
    df["Characteristic Length"].plot.hist(bins=40)
    plt.xscale("log")
    plt.savefig("len.png")
    print(df["Characteristic Length"].mean())
    print(df["A/M"].mean())
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
    df.to_excel("df.xlsx")
    #df["vel_x"]=600.0
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D
    # Constants
    g = 9.81  # m/s^2, acceleration due to gravity
    rho = 1.225  # kg/m^3, air density at sea level
    Cd = 2 # Drag coefficient, assumed constant for simplicity
    dt = 0.001  # Timestep in seconds
    counter=0
    class DebrisSimulation:
        def __init__(self, dataframe):
            self.df = dataframe
            self.pbar = tqdm(total=10000)
            print(dataframe)
            self.list1=[600.0]
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.lines = [self.ax.plot([], [], [], 'o')[0] for _ in range(len(self.df))]
            # Set the limits for all axes
            self.ax.set_xlim([-70, 70])
            self.ax.set_ylim([-70, 70])
            self.ax.set_zlim([-70, 50])
            self.df[['pos_x', 'pos_y', 'pos_z']] = self.df[['pos_x', 'pos_y', 'pos_z']].astype(float)
        def update_position_velocity(self, dt):
            for index, fragment in self.df.iterrows():
                A = fragment['area']  # Cross-sectional area
                v = np.array([fragment['vel_x'], fragment['vel_y'], fragment['vel_z']])
                v_mag = np.linalg.norm(v)

                if v_mag > 0:
                    # Component-wise drag calculation
                    drag_coefficient = -0.5 * rho * Cd * A
                    Fd_x = drag_coefficient * v[0]**2 * np.sign(v[0])
                    Fd_y = drag_coefficient * v[1]**2 * np.sign(v[1])
                    Fd_z = drag_coefficient * v[2]**2 * np.sign(v[2])

                    Fd = np.array([Fd_x, Fd_y, Fd_z])  # Total drag force vector
                else:
                    Fd = np.array([0, 0, 0])  # No drag if there's no velocity

                Fg = np.array([0, 0, -fragment['mass'] * g])  # Gravitational force
                Fnet = Fd + Fg  # Net force on the fragment

                a = Fnet / fragment['mass']  # Acceleration due to net force
                # Update velocity
                self.df.loc[index, ['vel_x', 'vel_y', 'vel_z']] = v + a * dt
                self.list1.append((v + a * dt)[0])
                if np.linalg.norm(v + a * dt)>10000:
                    print(v + a * dt)
                    print(fragment)
                    plt.close()
                    plt.plot(self.list1)
                    plt.savefig("debug.png")
                    exit()
                
                #print(fragment)
                # Update position
                self.df.loc[index, ['pos_x', 'pos_y', 'pos_z']] = np.array([fragment['pos_x'], fragment['pos_y'], fragment['pos_z']]) + v * dt + 0.5 * a * dt**2
                        
        def init(self):
            for line in self.lines:
                line.set_data([], [])
                line.set_3d_properties([])
            return self.lines

        def animate(self, i):
            self.update_position_velocity(dt)
            self.pbar.update(1)
            for line, (_, fragment) in zip(self.lines, self.df.iterrows()):
                # Convert the DataFrame slice to a numpy array explicitly
                pos_x = fragment['pos_x']
                pos_y = fragment['pos_y']
                line.set_data([pos_x], [pos_y])
                # Ensure z data is also set correctly
                line.set_3d_properties([fragment['pos_z']])
            return self.lines

        def run(self):
            ani = FuncAnimation(self.fig, self.animate, frames=np.arange(0, 10000), init_func=self.init, blit=True)
            return ani

    # Sample DataFrame setup
    #columns = ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', 'mass', 'area_to_mass']
    #data = np.random.rand(10, len(columns))  # Example data
    #df = pd.DataFrame(data, columns=columns)

    # Initialize and run the simulation
    simulation = DebrisSimulation(df)
    ani = simulation.run()

    # Save the animation
    ani.save('debris_trajectories.mp4', writer='ffmpeg', fps=1000)
    print(max(abs(df["pos_x"])))
if __name__=="__main__":
    main("config.json")