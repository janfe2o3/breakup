from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# Constants
g = 9.81  # m/s^2
rho = 1.225  # kg/m^3
Cd = 2 # Drag coefficient
dt = 0.01  # Timestep in seconds
counter=0


class TrajectorySimulation:
    def __init__(self, dataframe):
        self.df = dataframe
        self.pbar = tqdm(total=1000)
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
        ani = FuncAnimation(self.fig, self.animate, frames=np.arange(0, 1000), init_func=self.init, blit=True)
        return ani