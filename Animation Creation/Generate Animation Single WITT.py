#%% import libraries etc
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

#%% Code details
# This script is designed to read in data saved from 'Single WITT Simulations v2.0.py' and produce an animation of the single WITT system using it, created 15/08

#%% wave number function
def find_kw(v, h, g = 9.81, tol=1e-12, maxiter=50):
    v = np.atleast_1d(v).astype(float).flatten()   # handle scalars or arrays
    k_values = np.zeros_like(v, dtype=float)
    for j in range(len(v)):
        omega = 2*np.pi*v[j] # convert frequency to rad
        k0 = omega**2/g # wave number in deep water
        x0 = np.sqrt(k0*h) # initial guess of kh
        
        # iterate solution
        for i in range(maxiter):
            arg = (x0/np.cosh(x0))**2
            kh = x0*((k0*h + (arg) )/(x0*np.tanh(x0) + arg))
            if abs(x0-kh) < tol:
                # print(f'iterations used: {i}')
                break # kh/h
            x0 = kh
        k_values[j] = kh / h
    
    return k_values
#%% read in data from simulations

# define folder where the data is saved
output_folder = "02 Modelling/03 Summer Delivery/Results/Single WITT/Optimised system moored"
moving_frame = False

# load each dataset
sols_data = np.load(os.path.join(output_folder, "sim_outputs.npz"), allow_pickle=True)
Z0s_data = np.load(os.path.join(output_folder, "Z0s.npz"))
cairns_data = np.load(os.path.join(output_folder, "cairns_terms.npz"))
PTO_data = np.load(os.path.join(output_folder, "PTO_terms.npz"))
WITT_data = np.load(os.path.join(output_folder, "WITT_params.npz"))
wave_data = np.load(os.path.join(output_folder, "wave_params.npz"))
mooring_data = np.load(os.path.join(output_folder, "mooring_params.npz"))

# extract arrays
sols_list = sols_data["sols_3d"]
Z0s = Z0s_data["Z0s"]
cairns_terms = cairns_data["cairns_terms"]
PTO_terms = PTO_data["PTO_terms"]
WITT_params = WITT_data["WITT_params"]
mooring_params = mooring_data["mooring_params"]

# quick check
print("sols_3d shape:", sols_list.shape)
print("Z0s shape:", Z0s.shape)
print("cairns_terms shape:", cairns_terms.shape)
print("PTO_terms shape:", PTO_terms.shape)

# #%% generate animation
# pick which run to animate
run_index = 1
run_data = sols_list[run_index]   # shape should be (n_states, n_time_points)
Z0 = Z0s[run_index]

# define indices for x and z etc: [x, dx, z, dz, theta, dtheta, phi, dphi, theta+phi] --> 0, 1, 2, 3, 4, 5, 6, 7, 8
x_idx = 0
z_idx = 2
phi_idx = 6
thetaphi_idx = 8

# positional data
time_lim = 200
t_step = 0.1
time_idx = run_data.t <= time_lim
x = run_data.y[x_idx][time_idx]
z = run_data.y[z_idx][time_idx]
z = z-Z0
phi = run_data.y[phi_idx][time_idx]
thetaphi = run_data.y[thetaphi_idx][time_idx]
t = run_data.t[time_idx]

# WITT data
l = WITT_params[1]
D = WITT_params[0]
R = D/2

# wave data
A = cairns_terms[run_index,0,:]
sigma = cairns_terms[run_index,1,:]
n_waves = wave_data['n_waves']
a = wave_data['a']
v = np.array([0.3]) #wave_data['v']
phis = wave_data['phis']
h = wave_data['h']
k_w = find_kw(v, h) #(2 * np.pi * v) ** 2 / g #np.sqrt((2 * np.pi * v) ** 2 / g)
lambdas = 2 * np.pi / k_w

# mooring data
X0 = mooring_params[0]
alpha_m = mooring_params[1]
n_mooring = mooring_params[2]
h10 = h - R*np.cos(alpha_m) - Z0
h20 = h10*np.tan(X0)
xm_attach = R*np.sin(alpha_m)
zm_attach = R*np.cos(alpha_m)
xm1_end = -h20-xm_attach
xm2_end = h20+xm_attach
zm1_end = -h10-zm_attach
zm2_end = -h10-zm_attach

# set up axes etc
fig, ax = plt.subplots()
plot_margin = R*1.1
x_wave = []
if not moving_frame:
    ax.set_xlim(min(x)-plot_margin, max(x)+plot_margin) #min(x)-plot_margin
    ax.set_ylim(min(z)-plot_margin, max(z)+plot_margin)
    x_wave = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 500) # create a uniform x-array for the wave (independent of the buoy)
ax.set_xlabel('Horizontal position, x (m)')
ax.set_ylabel('Depth, z-Z0 (m)')
# ax.set_title(f'WITT Motion (Simulation {run_index})')
ax.set_aspect('equal', 'box')  # 'box' ensures the axes fit in the figure

# initialise plot elements
point, = ax.plot([], [], 'ko', markersize=3)
path, = ax.plot([], [], 'b-', lw=1)
pendulum_line, = ax.plot([], [], 'r-', lw=1.5)
pendulum_bob, = ax.plot([], [], 'ro', markersize=4)
ballast_bob, = ax.plot([], [], 'ko', markersize=4)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
circle = patches.Circle((x[0], z[0]), R, facecolor=(0, 0, 0, 0.2), edgecolor='black')
ax.add_patch(circle)
wave_line, = ax.plot([], [], 'k--', lw=1.5)  # cyan line for local wave
wave_curve, = ax.plot([], [], 'c-', lw=1)  # cyan dashed line for wave
mooring_line1, = ax.plot([], [], color='tab:orange', lw=1.5)
mooring_line2, = ax.plot([], [], color='tab:orange', lw=1.5)


# initialise animation path
x_path, z_path = [], []

def update(frame):
    # update buoy position
    x_path.append(x[frame])
    z_path.append(z[frame])
    point.set_data([x[frame]], [z[frame]])
    path.set_data(x_path, z_path)
    
    # Update axes to follow buoy
    if moving_frame:
        x_center, z_center = x[frame], z[frame]
        ax.set_xlim(x_center - 4*R, x_center + 4*R)
        ax.set_ylim(z_center - 4*R, z_center + 4*R)
        ax.figure.canvas.draw_idle()
        x_wave_local = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 500) # create a uniform x-array for the wave (independent of the buoy)
    else:
        x_wave_local = x_wave
    
    # update circle
    circle.center = (x[frame], z[frame])
    
    # update ballast
    x_ballast = x[frame] + R * np.sin(phi[frame])
    z_ballast = z[frame] - R * np.cos(phi[frame])
    ballast_bob.set_data([x_ballast],[z_ballast])
    
    # update pendulum
    x_bob = x[frame] + l * np.sin(thetaphi[frame])
    z_bob = z[frame] - l * np.cos(thetaphi[frame])
    pendulum_line.set_data([x[frame], x_bob],[z[frame], z_bob])
    pendulum_bob.set_data([x_bob], [z_bob])
    
    # update time text
    time_text.set_text(f"t = {t[frame]:.2f} s")
    
    # update local wave line
    x0 = x[frame]
    z0 = A[frame] # center of the line
    angle = sigma[frame]      # in radians
    L = R*1.5 #2.0  # half-length of the line, adjust as needed
    # compute endpoints
    dx = L * np.cos(angle)
    dz = L * np.sin(angle)
    x_start, x_end = x0 - dx, x0 + dx
    z_start, z_end = z0 - dz, z0 + dz
    wave_line.set_data([x_start, x_end], [z_start, z_end])
    
    # update wave curve
    t_frame = t[frame]
    A_list = []
    for x_temp in x_wave_local:
        A_temp = np.zeros(n_waves)
        for i in range(n_waves):
            A_temp[i] = a[i] * np.sin(2*np.pi * (v[i]*t_frame - x_temp/lambdas[i]) + phis[i])
        A_list.append(np.sum(A_temp))
    wave_curve.set_data(x_wave_local, A_list)
    
    # update mooring lines
    xm1 = x[frame] - xm_attach + R*np.sin(alpha_m) - R*np.cos(np.pi/2 - alpha_m + phi[frame])
    xm2 = x[frame] + xm_attach + R*np.sin(phi[frame]+alpha_m) - R*np.sin(alpha_m)
    zm1 = z[frame] - zm_attach - (R*np.cos(alpha_m-phi[frame]) - R*np.cos(alpha_m))
    zm2 = z[frame] - zm_attach + (R*np.cos(alpha_m) - R*np.cos(alpha_m+phi[frame]))
    if n_mooring != 0:
        mooring_line1.set_data([xm1, xm1_end],[zm1, zm1_end])
        mooring_line2.set_data([xm2, xm2_end],[zm2, zm2_end])
    
    if moving_frame:
        return ()
    else:
        return point, path, circle, ballast_bob, pendulum_line, pendulum_bob, time_text, wave_line, wave_curve, mooring_line1, mooring_line2

# create animation
speedup = 1   # how many times faster than real time
interval = 1000 * t_step / speedup  # convert to ms per frame
fps = 1000 / interval  # frames per second
ani = FuncAnimation(fig, update, frames=len(t), interval=interval, blit=False)

# save animation
print(f'Frequency is set to {v}, and run number set to {run_index}')
# output_folder = "02 Modelling/03 Summer Delivery/Animations/Single WITT"  # set folder path
# filename = "Optimised_moored_0.3Hz_200s.mp4"
# print(f"Saving animation to '{output_folder}' as '{filename}' ")
# ani.save(os.path.join(output_folder, filename), writer='ffmpeg', fps=fps,dpi=300)

# display animation
plt.show()
