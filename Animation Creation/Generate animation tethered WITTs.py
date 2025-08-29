#%% import libraries etc
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

#%% Code details
# This script is designed to read in data saved from 'Tethered WITT Simulations v1.0.py' and produce an animation of the tethered WITT system using it
# Created 17/08 using 'Generate animation single WITT.py'
#
# NOTE:
#       Updated animation to plot both WITTs, 17/08
#       Added a line for the tether between the buoys, which changes colour for taut vs slack, 17/08
#       Updated WITT_params to include tether natural length L0, 17/08

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
output_folder = "02 Modelling/03 Summer Delivery/Results/Tethered WITTs/Baseline Nm=1 Bilinear 150"

# load each dataset
sols_data = np.load(os.path.join(output_folder, "sim_outputs.npz"), allow_pickle=True)
Z0s_data = np.load(os.path.join(output_folder, "Z0s.npz"))
cairns_data = np.load(os.path.join(output_folder, "cairns_terms.npz"))
PTO_data = np.load(os.path.join(output_folder, "PTO_terms.npz"))
WITT_data = np.load(os.path.join(output_folder, "WITT_params.npz"))
wave_data = np.load(os.path.join(output_folder, "wave_params.npz"))
mooring_data = np.load(os.path.join(output_folder, "mooring_params.npz"))
tether_data = np.load(os.path.join(output_folder, "tether_params.npz"))

# extract arrays
sols_list = sols_data["sols_3d"]
Z0s = Z0s_data["Z0s"]
cairns_terms = cairns_data["cairns_terms"]
PTO_terms = PTO_data["PTO_terms"]
WITT_params = WITT_data["WITT_params"]
mooring_params = mooring_data["mooring_params"]
L0=tether_data['L0']
k0_tether=tether_data['k0']
k_tether=tether_data['k']
beta=tether_data['beta']
starting_separation=tether_data['separation']
tether_type=tether_data['type']

# quick check
print("sols_3d shape:", sols_list.shape)
print("Z0s shape:", Z0s.shape)
print("cairns_terms shape:", cairns_terms.shape)
print("PTO_terms shape:", PTO_terms.shape)

# #%% generate animation
# pick which run to animate
run_index = 2
run_data = sols_list[run_index]   # shape should be (n_states, n_time_points)
Z0 = Z0s[run_index]

# define indices for x and z etc: [x, dx, z, dz, theta, dtheta, phi, dphi] x2, [theta+phi]x2 --> 0:7, 8:15, 16:17
x1_idx = 0
z1_idx = 2
phi1_idx = 6
x2_idx = x1_idx+8
z2_idx = z1_idx+8
phi2_idx = phi1_idx+8
thetaphi1_idx = 16
thetaphi2_idx = 17

# positional data
x1 = run_data.y[x1_idx]
z1 = run_data.y[z1_idx]
z1 = z1-Z0
phi1 = run_data.y[phi1_idx]
x2 = run_data.y[x2_idx]
z2 = run_data.y[z2_idx]
z2 = z2-Z0
phi2 = run_data.y[phi2_idx]
thetaphi1 = run_data.y[thetaphi1_idx]
thetaphi2 = run_data.y[thetaphi2_idx]
t = run_data.t

# WITT data
l = WITT_params[1]
D = WITT_params[0]
R = D/2

# wave data
A1 = cairns_terms[run_index,0,:]
sigma1 = cairns_terms[run_index,1,:]
A2 = cairns_terms[run_index,2,:]
sigma2 = cairns_terms[run_index,3,:]
n_waves = wave_data['n_waves']
a = wave_data['a']
v = np.array([0.44]) #wave_data['v']
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
xm1_end = x1[0]-h20-xm_attach
xm2_end = x2[0]+h20+xm_attach
zm1_end = z1[0]-h10-zm_attach
zm2_end = z2[0]-h10-zm_attach

# set up axes etc
fig, ax = plt.subplots()
plot_margin = R*1.1
ax.set_xlim(min(x1)-plot_margin, max(x2)+plot_margin)
ax.set_ylim(min(min(z1),min(z2))-plot_margin, max(max(z1),max(z2))+plot_margin)
ax.set_xlabel('Horizontal position, x (m)')
ax.set_ylabel('Depth, z-Z0 (m)')
ax.set_title(f'WITT Motion (Simulation {run_index})')
ax.set_aspect('equal', 'box')  # 'box' ensures the axes fit in the figure

# initialise lines etc for animation
# --- WITT buoy 1 ---
point1, = ax.plot([], [], 'ko', markersize=3)
path1, = ax.plot([], [], 'b-', lw=1)
pendulum_line1, = ax.plot([], [], 'r-', lw=1.5)
pendulum_bob1, = ax.plot([], [], 'ro', markersize=4)
ballast_bob1, = ax.plot([], [], 'ko', markersize=4)
circle1 = patches.Circle((x1[0], z1[0]), R, facecolor=(0, 0, 0, 0.2), edgecolor='black')
ax.add_patch(circle1)
wave_line1, = ax.plot([], [], 'k--', lw=1.5)  # cyan line for local wave
x1_path, z1_path = [], []
# --- WITT buoy 2 ---
point2, = ax.plot([], [], 'ko', markersize=3)
path2, = ax.plot([], [], 'b-', lw=1)
pendulum_line2, = ax.plot([], [], 'r-', lw=1.5)
pendulum_bob2, = ax.plot([], [], 'ro', markersize=4)
ballast_bob2, = ax.plot([], [], 'ko', markersize=4)
circle2 = patches.Circle((x2[0], z2[0]), R, facecolor=(0, 0, 0, 0.2), edgecolor='black')
ax.add_patch(circle2)
wave_line2, = ax.plot([], [], 'k--', lw=1.5)  # cyan line for local wave
x2_path, z2_path = [], []
# --- time text ---
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
# --- wave forcing ---
wave_curve, = ax.plot([], [], 'c-', lw=1)  # cyan dashed line for wave
x_wave = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 500) # create a uniform x-array for the wave (independent of the buoy)
# --- tether ---
tether_line, = ax.plot([], [], 'k-', lw=1.5)#, zorder=10)
# --- mooring lines ---
mooring_line1, = ax.plot([], [], color='tab:orange', lw=1.5)
mooring_line2, = ax.plot([], [], color='tab:orange', lw=1.5)

def update(frame):
    # --- WITT buoy 1 ---
    # update buoy position
    x1_path.append(x1[frame])
    z1_path.append(z1[frame])
    point1.set_data([x1[frame]], [z1[frame]])
    path1.set_data(x1_path, z1_path)
    
    # update circle
    circle1.center = (x1[frame], z1[frame])
    
    # update ballast
    x1_ballast = x1[frame] + R * np.sin(phi1[frame])
    z1_ballast = z1[frame] - R * np.cos(phi1[frame])
    ballast_bob1.set_data([x1_ballast],[z1_ballast])
    
    # update pendulum
    x1_bob = x1[frame] + l * np.sin(thetaphi1[frame])
    z1_bob = z1[frame] - l * np.cos(thetaphi1[frame])
    pendulum_line1.set_data([x1[frame], x1_bob],[z1[frame], z1_bob])
    pendulum_bob1.set_data([x1_bob], [z1_bob])
    
    # update local wave line
    x10 = x1[frame]
    z10 = A1[frame] # center of the line
    angle1 = sigma1[frame]      # in radians
    L = R*1.5 #2.0  # half-length of the line, adjust as needed
    # compute endpoints
    dx1 = L * np.cos(angle1)
    dz1 = L * np.sin(angle1)
    x1_start, x1_end = x10 - dx1, x10 + dx1
    z1_start, z1_end = z10 - dz1, z10 + dz1
    wave_line1.set_data([x1_start, x1_end], [z1_start, z1_end])
    
    # --- WITT buoy 2 ---
    # update buoy position
    x2_path.append(x2[frame])
    z2_path.append(z2[frame])
    point2.set_data([x2[frame]], [z2[frame]])
    path2.set_data(x2_path, z2_path)
    
    # update circle
    circle2.center = (x2[frame], z2[frame])
    
    # update ballast
    x2_ballast = x2[frame] + R * np.sin(phi2[frame])
    z2_ballast = z2[frame] - R * np.cos(phi2[frame])
    ballast_bob2.set_data([x2_ballast],[z2_ballast])
    
    # update pendulum
    x2_bob = x2[frame] + l * np.sin(thetaphi2[frame])
    z2_bob = z2[frame] - l * np.cos(thetaphi2[frame])
    pendulum_line2.set_data([x2[frame], x2_bob],[z2[frame], z2_bob])
    pendulum_bob2.set_data([x2_bob], [z2_bob])
    
    # update local wave line
    x20 = x2[frame]
    z20 = A2[frame] # center of the line
    angle2 = sigma2[frame]      # in radians
    # compute endpoints
    dx2 = L * np.cos(angle2)
    dz2 = L * np.sin(angle2)
    x2_start, x2_end = x20 - dx2, x20 + dx2
    z2_start, z2_end = z20 - dz2, z20 + dz2
    wave_line2.set_data([x2_start, x2_end], [z2_start, z2_end])
    
    # --- time indicator ---
    # update time text
    time_text.set_text(f"t = {t[frame]:.2f} s")
    
    # --- wave forcing ---
    # update wave curve
    t_frame = t[frame]
    A_list = []
    for x_temp in x_wave:
        A_temp = np.zeros(n_waves)
        for i in range(n_waves):
            A_temp[i] = a[i] * np.sin(2*np.pi * (v[i]*t_frame - x_temp/lambdas[i]) + phis[i])
        A_list.append(np.sum(A_temp))
    wave_curve.set_data(x_wave, A_list)
    
    # --- tether line ---
    # plot tether as a line between buoys
    x1_tether = x1[frame] + R*np.sin(beta+phi1[frame])
    z1_tether = z1[frame] - R*np.cos(beta+phi1[frame]) # + R*np.sin(angle1)
    x2_tether = x2[frame] + R*np.sin(-beta+phi2[frame]) # - R*np.cos(angle2)
    z2_tether = z2[frame] - R*np.cos(-beta+phi2[frame]) # - R*np.sin(angle2)
    tether_line.set_data([x1_tether, x2_tether],[z1_tether, z2_tether])
    
    # compute length
    L = np.sqrt((x2_tether - x1_tether)**2 + (z2_tether - z1_tether)**2)

    # change colour depending on extension
    if L > L0:
        tether_line.set_color("red")   # taut
    else:
        tether_line.set_color("k") # slack
    
    # --- mooring lines ---
    xm1 = x1[frame] - xm_attach + R*np.sin(alpha_m) - R*np.cos(np.pi/2 - alpha_m + phi1[frame])
    xm2 = x2[frame] + xm_attach + R*np.sin(phi2[frame]+alpha_m) - R*np.sin(alpha_m)
    zm1 = z1[frame] - zm_attach - (R*np.cos(alpha_m-phi1[frame]) - R*np.cos(alpha_m))
    zm2 = z2[frame] - zm_attach + (R*np.cos(alpha_m) - R*np.cos(alpha_m+phi2[frame]))
    if n_mooring != 0:
        mooring_line1.set_data([xm1, xm1_end],[zm1, zm1_end])
        mooring_line2.set_data([xm2, xm2_end],[zm2, zm2_end])
    return point1, path1, circle1, ballast_bob1, pendulum_line1, pendulum_bob1, wave_line1, point2, path2, circle2, ballast_bob2, pendulum_line2, pendulum_bob2, wave_line2, time_text, wave_curve, tether_line, mooring_line1, mooring_line2

# create animation
t_step = 0.1  # seconds per solver step
speedup = 0.2  # how many times faster than real time
interval = 1000 * t_step / speedup  # convert to ms per frame
fps = 1000 / interval  # frames per second
ani = FuncAnimation(fig, update, frames=len(t), interval=interval, blit=True)

# save animation
print(f'Frequency is set to {v}, and run number set to {run_index}')
# output_folder = "02 Modelling/03 Summer Delivery/Animations/Tethered WITTs"  # set folder path
# filename = "baseline_tethered_0.44Hz.mp4"
# print(f"Saving animation to '{output_folder}' as '{filename}' ")
# ani.save(os.path.join(output_folder, filename), writer='ffmpeg', fps=fps)

# display animation
plt.show()