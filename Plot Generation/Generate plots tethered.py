#%% import libraries etc
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.integrate import trapezoid

#%% read in data from simulations

# define folder where the data is saved
output_folder = "02 Modelling/03 Summer Delivery/Results/Tethered WITTs/Bilinear/Phase Diffs Low k0(=20) nm=1"

# load each dataset
sols_data = np.load(os.path.join(output_folder, "sim_outputs.npz"), allow_pickle=True)
Z0s_data = np.load(os.path.join(output_folder, "Z0s.npz"))
cairns_data = np.load(os.path.join(output_folder, "cairns_terms.npz"))
PTO_data = np.load(os.path.join(output_folder, "PTO_terms.npz"))
WITT_data = np.load(os.path.join(output_folder, "WITT_params.npz"))
wave_data = np.load(os.path.join(output_folder, "wave_params.npz"))
loop_data = np.load(os.path.join(output_folder, "loop_params.npz"))
freq_data = np.load(os.path.join(output_folder, "freq_results.npz"))
mooring_data  = np.load(os.path.join(output_folder, "mooring_params.npz"))

# extract arrays
sols = sols_data["sols_3d"]
Z0s = Z0s_data["Z0s"]
cairns_terms = cairns_data["cairns_terms"]
PTO_terms = PTO_data["PTO_terms"]
WITT_params = WITT_data["WITT_params"]
loop_params = loop_data["loop_params"]
sols_freqs = freq_data["freqs"]
sols_amplitudes = freq_data["amps"]
# print(Z0s)

# quick check
print("sols_3d shape:", sols.shape)
print("Z0s shape:", Z0s.shape)
print("cairns_terms shape:", cairns_terms.shape)
print("PTO_terms shape:", PTO_terms.shape)

#%% setup other variables
# setup time parameters
t_start = 0
# t_current = t_start
t_end = 200
# t_step = 0.1
# t_eval = np.arange(t_start, t_end, t_step)
# t_span = (t_start, t_end)

plot_labels = loop_params

#%% calculate energy
E_PTO1 = []
E_PTO2 = []
for i,sol in enumerate(sols):
    Power_vals = PTO_terms[i]
    E_PTO1.append(trapezoid(Power_vals[2,:], sol.t))  # or cumtrapz for time-resolved energy
    E_PTO2.append(trapezoid(Power_vals[2+3,:], sol.t))  # or cumtrapz for time-resolved energy
E_PTO1 = np.array(E_PTO1)
E_PTO2 = np.array(E_PTO2)
P_ave1 = E_PTO1/200
P_ave2 = E_PTO2/200
print(E_PTO1, E_PTO2)


#%% initial plots to check results

# plot dtheta as a test for PTO threshold
plt.figure(figsize=(8, 5))
var_id = 5
for i,sol in enumerate(sols):
    label = plot_labels[i]
    plt.plot(sol.t, sol.y[var_id], label=f'{label}, WITT 1') #, label='n_mooring = 2')
    plt.plot(sol.t, sol.y[var_id+8], label=f'{label}, WITT 2') #, label='n_mooring = 2')
plt.title('Pendulum angular velocity')
plt.ylabel('dtheta (rad/s)')
plt.xlabel('Time (s)')
plt.grid()
plt.legend()
plt.xlim([t_start, t_end])
plt.tight_layout()
plt.show()

# plot x, z
plt.figure(figsize=(8, 5))
plt.subplot(2, 1, 1)
var_id = 0
for i,sol in enumerate(sols):
    label = plot_labels[i]
    plt.plot(sol.t, sol.y[var_id], label=f'{label}, WITT 1') #, label='n_mooring = 2')
    plt.plot(sol.t, sol.y[var_id+8], label=f'{label}, WITT 2') #, label='n_mooring = 2')
    # plt.plot(sol_single_m1.t, sol_single_m1.y[var_id]) #, label='n_mooring = 1')
    # plt.plot(sol_single_m0.t, sol_single_m0.y[var_id]) #, label='n_mooring = 0')
plt.title('Horizontal position')
plt.ylabel('x (m)')
plt.xlabel('Time (s)')
plt.grid()
plt.legend()
plt.xlim([t_start, t_end])
plt.subplot(2, 1, 2)
var_id = 2
for i,sol in enumerate(sols):
    label = plot_labels[i]
    plt.plot(sol.t, sol.y[var_id], label=f'{label}, WITT 1') #, label='n_mooring = 2')
    plt.plot(sol.t, sol.y[var_id+8], label=f'{label}, WITT 2') #, label='n_mooring = 2')
# plt.plot(sol_single_m2.t, sol_single_m2.y[var_id], label='n_mooring = 2')
# plt.plot(sol_single_m1.t, sol_single_m1.y[var_id], label='n_mooring = 1')
# plt.plot(sol_single_m0.t, sol_single_m0.y[var_id], label='n_mooring = 0')
plt.title('Vertical position')
plt.ylabel('z (m)')
plt.xlabel('Time (s)')
plt.grid()
plt.xlim([t_start, t_end])
plt.tight_layout()
# plt.savefig('02 Modelling/03 Summer Delivery/Single WITT - x z.pdf')
plt.show()

# plot depth=z-Z0, wave amplitude 
plt.figure(figsize=(8, 5))
plt.subplot(2, 1, 1)
var_id = 2
for i,sol in enumerate(sols):
    label = plot_labels[i]
    plt.plot(sol.t, sol.y[var_id]-Z0s[i], label=f'{label}, WITT 1') #, label='n_mooring = 2')
    plt.plot(sol.t, sol.y[var_id+8]-Z0s[i], label=f'{label}, WITT 2') #, label='n_mooring = 2')
# plt.plot(sol_single_m2.t, sol_single_m2.y[var_id], label='n_mooring = 2')
# plt.plot(sol_single_m1.t, sol_single_m1.y[var_id], label='n_mooring = 1')
# plt.plot(sol_single_m0.t, sol_single_m0.y[var_id], label='n_mooring = 0')
plt.title('Vertical position')
plt.ylabel('Depth (z-Z0) (m)')
plt.xlabel('Time (s)')
plt.legend()
plt.grid()
plt.xlim([t_start, t_end])
plt.subplot(2, 1, 2)
for i,sol in enumerate(sols):
    label = plot_labels[i]
    plt.plot(sol.t, cairns_terms[i][0,:], label=f'{label}, WITT 1') #, label='n_mooring = 2')
    plt.plot(sol.t, cairns_terms[i][0+2,:], label=f'{label}, WITT 2') #, label='n_mooring = 2')
# plt.plot(sol_single_m2.t, sol_single_m2.y[var_id], label='n_mooring = 2')
# plt.plot(sol_single_m1.t, sol_single_m1.y[var_id], label='n_mooring = 1')
# plt.plot(sol_single_m0.t, sol_single_m0.y[var_id], label='n_mooring = 0')
plt.title('Wave amplitude')
plt.ylabel('A(x,t) (m)')
plt.xlabel('Time (s)')
plt.grid()
plt.xlim([t_start, t_end])
plt.tight_layout()
# plt.savefig('02 Modelling/03 Summer Delivery/Single WITT - x z.pdf')
plt.show()

# plot theta, theta+phi, phi
plt.figure(figsize=(8, 5))
plt.subplot(3, 1, 1)
var_id = 4
for i,sol in enumerate(sols):
    label = plot_labels[i]
    # time_temp, angle_temp = wrap_angles(sol.t, sol.y[var_id])  # wrap angles to [-pi, pi]
    # plt.plot(time_temp, angle_temp, label=label)
    plt.plot(sol.t, sol.y[var_id], label=f'{label}, WITT 1') #, label='n_mooring = 2')
    plt.plot(sol.t, sol.y[var_id+8], label=f'{label}, WITT 2') #, label='n_mooring = 2')
# plt.plot(sol_single_m2.t, sol_single_m2.y[var_id] + sol_single_m2.y[var_id+2], label='n_mooring = 2')
# plt.plot(sol_single_m1.t, sol_single_m1.y[var_id] + sol_single_m1.y[var_id+2], label='n_mooring = 1')
# plt.plot(sol_single_m0.t, sol_single_m0.y[var_id] + sol_single_m0.y[var_id+2], label='n_mooring = 0')
plt.title('Pendulum angle (local axes)')
plt.ylabel('theta (rad)')
plt.xlabel('Time (s)')
plt.grid()
plt.legend()
plt.xlim([t_start, t_end])
plt.subplot(3, 1, 2)
var_id = 16
for i,sol in enumerate(sols):
    label = plot_labels[i]
    # time_temp, angle_temp = wrap_angles(sol.t, sol.y[var_id])  # wrap angles to [-pi, pi]
    # plt.plot(time_temp, angle_temp, label=label)
    plt.plot(sol.t, sol.y[var_id], label=f'{label}, WITT 1') #, label='n_mooring = 2')
    plt.plot(sol.t, sol.y[var_id+1], label=f'{label}, WITT 2') #, label='n_mooring = 2')
# plt.plot(sol_single_m2.t, sol_single_m2.y[var_id] + sol_single_m2.y[var_id+2], label='n_mooring = 2')
# plt.plot(sol_single_m1.t, sol_single_m1.y[var_id] + sol_single_m1.y[var_id+2], label='n_mooring = 1')
# plt.plot(sol_single_m0.t, sol_single_m0.y[var_id] + sol_single_m0.y[var_id+2], label='n_mooring = 0')
plt.title('Pendulum angle (global axes)')
plt.ylabel('theta+phi (rad)')
plt.xlabel('Time (s)')
plt.grid()
plt.xlim([t_start, t_end])
plt.subplot(3, 1, 3)
var_id = 6
for i,sol in enumerate(sols):
    label = plot_labels[i]
    # time_temp, angle_temp = wrap_angles(sol.t, sol.y[var_id])  # wrap angles to [-pi, pi]
    # plt.plot(time_temp, angle_temp, label=label)
    plt.plot(sol.t, sol.y[var_id], label=f'{label}, WITT 1') #, label='n_mooring = 2')
    plt.plot(sol.t, sol.y[var_id+8], label=f'{label}, WITT 2') #, label='n_mooring = 2')
# plt.plot(sol_single_m2.t, sol_single_m2.y[var_id], label='n_mooring = 2')
# plt.plot(sol_single_m1.t, sol_single_m1.y[var_id], label='n_mooring = 1')
# plt.plot(sol_single_m0.t, sol_single_m0.y[var_id], label='n_mooring = 0')
plt.title('Buoy angle')
plt.ylabel('phi (rad)')
plt.xlabel('Time (s)')
plt.grid()
plt.xlim([t_start, t_end])
plt.tight_layout()
# plt.savefig('02 Modelling/03 Summer Delivery/Single WITT - theta phi.pdf')
plt.show()

# plot buoy angle, wave angle 
plt.figure(figsize=(8, 5))
plt.subplot(2, 1, 1)
var_id = 6
for i,sol in enumerate(sols):
    label = plot_labels[i]
    # time_temp, angle_temp = wrap_angles(sol.t, sol.y[var_id])  # wrap angles to [-pi, pi]
    # plt.plot(time_temp, angle_temp, label=label)
    plt.plot(sol.t, sol.y[var_id], label=f'{label}, WITT 1') #, label='n_mooring = 2')
    plt.plot(sol.t, sol.y[var_id+8], label=f'{label}, WITT 2') #, label='n_mooring = 2')
    # plt.plot(sol.t, cairns_terms[i][1,:], label='wave angle sigma(x,t)')
# plt.plot(sol_single_m2.t, sol_single_m2.y[var_id], label='n_mooring = 2')
# plt.plot(sol_single_m1.t, sol_single_m1.y[var_id], label='n_mooring = 1')
# plt.plot(sol_single_m0.t, sol_single_m0.y[var_id], label='n_mooring = 0')
plt.title('Buoy angle')
plt.ylabel('phi (rad)')
plt.xlabel('Time (s)')
plt.legend()
plt.grid()
plt.xlim([t_start, t_end])
plt.subplot(2, 1, 2)
for i,sol in enumerate(sols):
    label = plot_labels[i]
    plt.plot(sol.t, cairns_terms[i][1,:], label=f'{label}, WITT 1') #, label='n_mooring = 2')
    plt.plot(sol.t, cairns_terms[i][1+2,:], label=f'{label}, WITT 2') #, label='n_mooring = 2')
# plt.plot(sol_single_m2.t, sol_single_m2.y[var_id], label='n_mooring = 2')
# plt.plot(sol_single_m1.t, sol_single_m1.y[var_id], label='n_mooring = 1')
# plt.plot(sol_single_m0.t, sol_single_m0.y[var_id], label='n_mooring = 0')
plt.title('Wave angle')
plt.ylabel('sigma(x,t) (m)')
plt.xlabel('Time (s)')
plt.grid()
plt.xlim([t_start, t_end])
plt.tight_layout()
# plt.savefig('02 Modelling/03 Summer Delivery/Single WITT - x z.pdf')
plt.show()

# plot frequency spectrum of x,z
plt.figure(figsize=(8, 5))
plt.subplot(2, 1, 1)
var_id = 0 # from state vectos (x, dx, z, dz, theta, dtheta, phi, dphi, theta+phi)
for i,freqs in enumerate(sols_freqs):
    label = plot_labels[i]
    plt.plot(freqs[var_id, :], sols_amplitudes[i][var_id, :], label=f'{label}, WITT 1')
    plt.plot(freqs[var_id+8, :], sols_amplitudes[i][var_id+8, :], label=f'{label}, WITT 2')
# plt.plot(sols_freqs[0][var_id, :], sols_amplitudes[0][var_id, :], label='n_mooring = 2')
# plt.plot(sols_freqs[1][var_id, :], sols_amplitudes[1][var_id, :], label='n_mooring = 1')
# plt.plot(sols_freqs[2][var_id, :], sols_amplitudes[2][var_id, :], label='n_mooring = 0')
plt.title('Horizontal position')
plt.ylabel('Amplitudes')
plt.grid()
plt.xlim([0, 2])
plt.legend()
plt.subplot(2, 1, 2)
var_id = 2
for i,freqs in enumerate(sols_freqs):
    label = plot_labels[i]
    plt.plot(freqs[var_id, :], sols_amplitudes[i][var_id, :], label=f'{label}, WITT 1')
    plt.plot(freqs[var_id+8, :], sols_amplitudes[i][var_id+8, :], label=f'{label}, WITT 2')
# plt.plot(sols_freqs[0][var_id, :], sols_amplitudes[0][var_id, :], label='n_mooring = 2')
# plt.plot(sols_freqs[1][var_id, :], sols_amplitudes[1][var_id, :], label='n_mooring = 1')
# plt.plot(sols_freqs[2][var_id, :], sols_amplitudes[2][var_id, :], label='n_mooring = 0')
plt.title('Vertical position')
plt.ylabel('Amplitudes')
plt.xlabel('Frequencies (Hz)')
plt.xlim([0, 2])
plt.grid()
plt.tight_layout()
# plt.savefig('02 Modelling/03 Summer Delivery/Single WITT - x z freqs.pdf')
plt.show()

# plot frequency spectrum of theta, theta+phi, phi
plt.figure(figsize=(8, 5))
plt.subplot(3, 1, 1)
var_id = 4
for i,freqs in enumerate(sols_freqs):
    label = plot_labels[i]
    plt.plot(freqs[var_id, :], sols_amplitudes[i][var_id, :], label=f'{label}, WITT 1')
    plt.plot(freqs[var_id+8, :], sols_amplitudes[i][var_id+8, :], label=f'{label}, WITT 2')
# plt.plot(sols_freqs[0][var_id, :], sols_amplitudes[0][var_id, :], label='n_mooring = 2')
# plt.plot(sols_freqs[1][var_id, :], sols_amplitudes[1][var_id, :], label='n_mooring = 1')
# plt.plot(sols_freqs[2][var_id, :], sols_amplitudes[2][var_id, :], label='n_mooring = 0')
plt.title('Pendulum angle, theta (local)')
plt.ylabel('Amplitudes')
plt.grid()
plt.xlim([0, 2])
plt.legend()
plt.subplot(3, 1, 2)
var_id = 16
for i,freqs in enumerate(sols_freqs):
    label = plot_labels[i]
    plt.plot(freqs[var_id, :], sols_amplitudes[i][var_id, :], label=f'{label}, WITT 1')
    plt.plot(freqs[var_id+1, :], sols_amplitudes[i][var_id+1, :], label=f'{label}, WITT 2')
# plt.plot(sols_freqs[0][var_id, :], sols_amplitudes[0][var_id, :], label='n_mooring = 2')
# plt.plot(sols_freqs[1][var_id, :], sols_amplitudes[1][var_id, :], label='n_mooring = 1')
# plt.plot(sols_freqs[2][var_id, :], sols_amplitudes[2][var_id, :], label='n_mooring = 0')
plt.title('Pendulum angle, theta+phi (global)')
plt.ylabel('Amplitudes')
plt.xlim([0, 2])
plt.grid()
plt.subplot(3, 1, 3)
var_id = 6
for i,freqs in enumerate(sols_freqs):
    label = plot_labels[i]
    plt.plot(freqs[var_id, :], sols_amplitudes[i][var_id, :], label=f'{label}, WITT 1')
    plt.plot(freqs[var_id+8, :], sols_amplitudes[i][var_id+8, :], label=f'{label}, WITT 2')
# plt.plot(sols_freqs[0][var_id, :], sols_amplitudes[0][var_id, :], label='n_mooring = 2')
# plt.plot(sols_freqs[1][var_id, :], sols_amplitudes[1][var_id, :], label='n_mooring = 1')
# plt.plot(sols_freqs[2][var_id, :], sols_amplitudes[2][var_id, :], label='n_mooring = 0')
plt.title('Buoy angle, phi')
plt.ylabel('Amplitudes')
plt.xlabel('Frequencies (Hz)')
plt.xlim([0, 2])
plt.grid()
plt.tight_layout()
# plt.savefig('02 Modelling/03 Summer Delivery/Single WITT - theta phi freqs.pdf')
plt.show()

# plot natural damping (friction), applied torque, power using PTO_terms = [friction, torque, power]
plt.figure(figsize=(8, 5))
plt.subplot(3, 1, 1) # plot friction torque
var_id = 0
for i,sol in enumerate(sols):
    label = plot_labels[i]
    plt.plot(sol.t, PTO_terms[i][var_id,:], label=f'{label} WITT 1')
    plt.plot(sol.t, PTO_terms[i][var_id+3,:], label=f'{label} WITT 2')
plt.title('Natural damping (friction etc)')
plt.ylabel('F theta (Nm)')
plt.xlabel('Time (s)')
plt.grid()
plt.legend()
plt.xlim([t_start, t_end])
plt.subplot(3, 1, 2) # plot PTO torque
var_id = 1
for i,sol in enumerate(sols):
    label = plot_labels[i]
    plt.plot(sol.t, PTO_terms[i][var_id,:], label=f'{label} WITT 1')
    plt.plot(sol.t, PTO_terms[i][var_id+3,:], label=f'{label} WITT 2')
plt.title('PTO damping torque')
plt.ylabel('Torque (Nm)')
plt.xlabel('Time (s)')
plt.grid()
plt.legend()
plt.xlim([t_start, t_end])
plt.subplot(3, 1, 3) # plot PTO power
var_id = 2
for i,sol in enumerate(sols):
    label = plot_labels[i]
    plt.plot(sol.t, PTO_terms[i][var_id,:], label=f'{label} WITT 1')
    plt.plot(sol.t, PTO_terms[i][var_id+3,:], label=f'{label} WITT 2')
plt.title('PTO power')
plt.ylabel('Power (W)')
plt.xlabel('Time (s)')
plt.grid()
plt.legend()
plt.xlim([t_start, t_end])
plt.tight_layout()
# plt.savefig('02 Modelling/03 Summer Delivery/Single WITT - theta phi.pdf')
plt.show()