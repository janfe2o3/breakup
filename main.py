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
            # Extract velocity components and compute magnitude
            velocities = self.df[['vel_x', 'vel_y', 'vel_z']].values
            v_mags = np.linalg.norm(velocities, axis=1)
            
            # Precompute gravitational force for all fragments
            Fg = np.array([0, 0, -g]) * self.df['mass'].values[:, np.newaxis]
            
            # Initialize drag force array
            Fd = np.zeros_like(velocities)
            
            # Calculate drag force only for fragments with non-zero velocity
            non_zero_vel = v_mags > 0
            A = self.df.loc[non_zero_vel, 'area'].values
            drag_coefficient = -0.5 * rho * Cd * A
            Fd_x = drag_coefficient * velocities[non_zero_vel, 0]**2 * np.sign(velocities[non_zero_vel, 0])
            Fd_y = drag_coefficient * velocities[non_zero_vel, 1]**2 * np.sign(velocities[non_zero_vel, 1])
            Fd_z = drag_coefficient * velocities[non_zero_vel, 2]**2 * np.sign(velocities[non_zero_vel, 2])
            Fd[non_zero_vel] = np.vstack((Fd_x, Fd_y, Fd_z)).T
            
            # Calculate net force
            Fnet = Fd + Fg
            
            # Calculate acceleration
            a = Fnet / self.df['mass'].values[:, np.newaxis]
            
            # Update velocity
            new_velocities = velocities + a * dt
            self.df[['vel_x', 'vel_y', 'vel_z']] = new_velocities
            
            # Update position
            new_positions = self.df[['pos_x', 'pos_y', 'pos_z']].values + velocities * dt + 0.5 * a * dt**2
            self.df[['pos_x', 'pos_y', 'pos_z']] = new_positions
                            
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
    ani.save('debris_trajectories2.mp4', writer='ffmpeg', fps=1000)
    print(max(abs(df["pos_x"])))
if __name__=="__main__":
    main("config.json")