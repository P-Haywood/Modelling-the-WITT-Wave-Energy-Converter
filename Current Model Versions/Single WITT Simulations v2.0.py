#%% import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.fft import fft, fftfreq
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.optimize import fsolve
import sys
from scipy.integrate import trapezoid
from scipy.integrate import quad


#%% version details
# v2.0 details:
# Created on 07/08 from v1.0, the main aim of this version is to clean up everything even further and remove old bits of code...
# ...in preperation for creating the final tethered system code
#
# Key things to NOTE:
#                 - created equilibrium_depth() and check_initial_mooring() functions to reduce repeated code, 07/08
#                 - removed code related to Morison's eqn hydrodynamic force, 07/08
#                 - put buoyancy_force() function back in use with a slight update (added D to the args provided), 07/08
#                 - reintroduced horizontal_hydrostatic_force() function with a slight update to ensure M_hydro is always defined (even if just as 0), 07/08
#                 - added drag_forces() function, 07/08
#                 - ensured mooring forces, non-conservative forces, and equations of motion etc are consistent, 07/08
#                 - updated logic at start of horizontal_hydrostatic_force() to prevent h_s > 2R or <-2R ((h_s-R)>R or (h_s-R)<-R) and added an error message in case it happens, 11/08
#                 - updated wave_forcing_cairns() to return average amplitude across buoy (same as sigma), 11/08
#                 - updated wavenumber calculation with find_kw() function which uses an iterative Newton-Raphson method to generalise to all depths, 11/08
#                 - updated simplest_mooring() Fz for n_mooring = 2 to use -T0 (previously had a typo and was using +T0), 11/08
#                 - updated simplest_mooring() for n_mooring = 1 scenario, NOTE: ONLY VALID for tethered system
#                 - updated drag_forces() to include a moment due to F_drag, acting through x_m (centre of waterline across buoy), values are tiny anyway, 11/08
#                 - changed drag torque to use tangential drag coefficient Cdt = 0.1, as per R Branch, 12/08
#                 - updated find_kw() to accept an array of frequencies values (v) and return an array of wave numbers (k_w), 13/08
#                 - added 'WITT_params', 'loop_params', 'wave_params', and 'freq_results' to saved outputs, 15/08
#                 - updated dtheta_PTO to 4.5rad/s as per motor torque constant on spec sheet provided by Martin Wickett, 17,08
#                 - updated geometric calcs of mooring attachment points in simple_taut and piecewise mooring functions, 18/08
#                 - updated z_a2 calc in mooring attachment point calcs (simple_taut etc) as was incorrect, 18/08
#                 - added saving of mooring_params (X0, alpha_m, n_mooring) for use in animations etc, 18/08
#                 - updated z drag force reference area so it's np.pi*R**2 for h_s-R>0 or 0 when above water, 21/08
#                 - removed cos(sigma)term from buoyancy force, as not sure it should actually apply, 21/08
#                 - moved Mmat to end and changed Cm to depend on submersion, i.e. Cm = 1 if not submerged, 21/08
#                 - corrected hydrostatic moment arm from (z_l-z_cl) to (z-z_cl), 21/08

# v1.0 details:
# Created on 23/07 using code from '2D WITT ODE v2.1.py', this code models a single WITT buoy system but in a neater and cleaner way...
# ...with things turned into functions, corrections made, old code elements omitted etc
#
# Key things to NOTE:
#                 - T0 is per mooring line, so vertical tension is n_mooring*T0*cos(X0)
#                 - equilibrium depth Z0 calc updated here to reflect initial vertical mooring tension, but not updated in development code!
#                 - fixed mooring force calculations (incorrect geometry for attachment points/mooring lines), 29/07
#                 - ***IMPORTANT: updated Fb to act upwards in dz equation, as believe it should, 30/07***
#                 - changed submerged height variable name from h to h_s so as not to get confused with water depth h in mooring calcs, 31/07
#                 - submerged height h_s constrained from 0 to D and resulting forces/moments etc updated to match, 31/07
#                 - added PTO on/off damping beyond a threshold angular velocity and relevant plots etc...
#                           ...however have to use a heaviside approximation (with epsilon around = 0.001) and LSODA for it to run, 31/07
#                 - corrected 'total_weight' term to (M+m)g, previously was just total mass, 31/07
#                 - added horizontal hydrodynamics forcing term due to difference in submerged frontal areas on left and right sides of buoy 05/08
#                 - added added mass coefficient, 05/08
#                 - added drag terms, 05/08
#                 - added lots of checks etc to prevent inf or NaN values crashing the code, 06/08
#                 - added event detection functionality for PTO force...
#                            ...can either run using Heaviside with LSODA, or event detection with LSODA, or event detection+Heaviside with Radau, 07/08
#                 - updated drag to also act in z and phi motions, 07/08
#                 - added an extra mooring_type - 'simplest_mooring' which just uses stiffnesses Kx, Kz, Kphi, Kxphi, 07/08
#                 - added code for saving data after running simulations, activated with a boolean flag, 07/08


#%% define functions

# function to run entire simulation
def run_sim(WITT_function, mooring_type, n_mooring, t_span, r0, t_eval, method, rtol, atol, env_params, witt_params, mooring_params):
    print(f'-------Running simulation for {WITT_function.__name__} with {n_mooring} {mooring_type} mooring lines------')
    
    # read in parameters from dictionaries
    g, rho_w, gamma_z, gamma_x, gamma_phi, a, v, phis, k_w, lambdas, n_waves = get_values_env(env_params)
    witt_size, m, m_s, m_pto, m_b, l, D, R, S, zeta0, c0, zeta, c, dtheta_PTO, thickness, M, J, total_weight, Ca, Cd  = get_values_witt(witt_params)
    h, alpha_m, X0, T0z, T0x, lambda_m0, lambda_m, slack_allowed, Kx, Kz, Kphi, Kxphi, Kzphi = get_values_mooring(mooring_params) # args: h, alpha_m, X0, T0z, T0x, lambda_m0, lambda_m, slack_allowed, Kx, Kz, Kphi, Kxphi, Kzphi
    t_start, t_end = t_span
    
    # calculate and print important system parameters
    buoy_r_gyration, f_nat_p, f_nat_b = calc_nat_freqs(env_params, witt_params)
    print_key_params(env_params, witt_params, f_nat_p, f_nat_b, buoy_r_gyration)
    max_Fb, max_mooring_Fz = check_can_float(env_params, witt_params)
    
    # initialise mooring
    if n_mooring == 0:
        print('No mooring lines...')
        mooring_params_calced = []
        Fzm_0 = 0
        
        # equilibrium depth
        Z0, eq_Fb, S_offset = equilibrium_depth(R, rho_w, g, total_weight, Fzm_0) # returns: Z0, eq_Fb, S_offset
        
    else:
        print(f'Initialising mooring with {n_mooring} lines...')
        
        if mooring_type == 'simplest_mooring':
            print('Using the simplest mooring model...')
            
            # check not using n_mooring == 1
            if n_mooring == 1:
                print(f'Error: using n_mooring = 1, but simplest_mooring is only valid for n_mooring = 1 when part of the tethered system.')
                sys.exit(1)
            
            # check T0z is sufficiently large
            if T0z < Kz*1:
                print(f'Warning: T0z is too low, if buoy moves past z={-T0z/Kz}m, Fzm will be 0')
            else:
                print(f'T0z is likely large enough to avoid Fzm = 0, which will occur at z={-T0z/Kz}m')
            
            # calculate equilibrium depth
            Fzm_0 = T0z
            T0 = T0z/(n_mooring*np.cos(X0))
            T0x = T0*np.sin(X0)
            Kx = Kz/(n_mooring*np.cos(X0)) * np.sin(X0) * n_mooring
            update_params_mooring(mooring_params,T0x=T0x)
            update_params_mooring(mooring_params,Kx=Kx)
            Z0, eq_Fb, S_offset = equilibrium_depth(R, rho_w, g, total_weight, Fzm_0)
            
            # print mooring parameters
            print (f'Mooring parameters: T0z {T0z}, T0x {T0x}, Kxx {Kx}, Kzz {Kz}, Kphi {Kphi}, Kxphi {Kxphi}')
            mooring_params_calced = [] # initialise empty list so code runs
            
            # calculate initial mooring forces
            Fx, Fz, Mphi = simplest_mooring(0, 0, 0, 0, T0z, T0x, Kx, Kz, Kphi, Kxphi, Kzphi, n_mooring) #simplest_mooring(x, z, phi, T0z, T0x, Kx, Kz, Kphi, Kxphi, Kzphi, n_mooring)
            print(f'Initial mooring forces: Fx {Fx}, Fz {Fz}, Mphi {Mphi}')
            
            # check vertical mooring force matches that from T0
            check_initial_mooring(Fz, Fzm_0, max_mooring_Fz)
            
        elif mooring_type == 'simple_taut':
            print('Using simple taut elastic mooring model...')
            
            # calculate equilibrium depth
            T0 = T0z/(n_mooring*np.cos(X0))
            Fzm_0 = T0z #n_mooring*T0*np.cos(X0) # initial vertical mooring force
            Z0, eq_Fb, S_offset = equilibrium_depth(R, rho_w, g, total_weight, Fzm_0)
            
            # calculate other parameters
            h1 = h - R*np.cos(alpha_m) - Z0 # initial depth of mooring point on hull, = water depth - mooring position from centre - Z0
            h2 = h1*np.tan(X0) # initial projected length of mooring line on horizontal plane (m)
            L0_prestressed = np.sqrt(h1**2 + h2**2) # initial pre-stressed length of mooring line
            L0_true = L0_prestressed - T0/lambda_m0 # calculate true natural length of mooring line based on tension and prestressed length
            mooring_params_calced = [h1, h2, L0_true]

            # print mooring parameters
            print (f'Mooring parameters: T0 {T0}, X0 (initial angle) {X0}, lambda_m0 {lambda_m0}, h1 {h1}, h2 {h2}, L0_prestressed {L0_prestressed}, L0_true {L0_true}')
            
            # calculate initial mooring forces
            Fx, Fz, Mphi = simple_taut_mooring(0, 0, 0, 0, R, n_mooring, Z0, alpha_m, h, lambda_m0, mooring_params_calced)  # initial conditions for mooring forces
            print(f'Initial mooring forces: Fx {Fx}, Fz {Fz}, Mphi {Mphi}')
            
            # check vertical mooring force matches that from T0
            check_initial_mooring(Fz, Fzm_0, max_mooring_Fz)

        elif mooring_type == 'piecewise':
            print('Using piecewise mooring model...')
            
            # calculate equilibrium depth
            T0 = T0z/(n_mooring*np.cos(X0))
            Fzm_0 = T0z #n_mooring*T0*np.cos(X0) # initial vertical mooring force
            Z0, eq_Fb, S_offset = equilibrium_depth(R, rho_w, g, total_weight, Fzm_0)
            
            # define parameters
            h1 = h - R*np.cos(alpha_m) - Z0 # initial depth of mooring point on hull, = water depth - mooring position from centre - Z0
            h2 = h1*np.tan(X0) # initial projected length of mooring line on horizontal plane (m)
            L0_prestressed = np.sqrt(h1**2 + h2**2) # initial pre-stressed length of mooring line
            L0_true = L0_prestressed - T0/lambda_m0 # calculate true natural length of mooring line based on prestressed tension and initial length
            slack_limit = L0_prestressed - L0_true + slack_allowed
            mooring_params_calced = [h1, h2, L0_true, slack_limit]
            
            # print mooring parameters
            print (f'Mooring parameters: lambda_m {lambda_m}, lambda_m0 {lambda_m0}, h1 {h1}, h2 {h2}, L0_prestressed {L0_prestressed}, L0_true {L0_true}, slack_limit {slack_limit}')
            
            # calculate initial mooring forces
            Fx, Fz, Mphi = piecewise_mooring(0, 0, 0, R, n_mooring, Z0, alpha_m, h, lambda_m0, lambda_m, mooring_params_calced)  # initial conditions for mooring forces
            print(f'Initial mooring forces: Fx {Fx}, Fz {Fz}, Mphi {Mphi}')
            
            # check vertical mooring force matches that from T0
            check_initial_mooring(Fz, Fzm_0, max_mooring_Fz)
    
    # solve ODEs
    print('Solving ODE...')
    if pto_calc == False:
        pto_flag = False
        sol = solve_ivp(WITT_function, t_span, r0, t_eval=t_eval, method=method, rtol=rtol, atol=atol, args=(mooring_type, n_mooring, Z0, mooring_params_calced, env_params, witt_params, mooring_params, pto_flag))
    else:
        if event_flag == False:
            pto_flag = False
            sol = solve_ivp(WITT_function, t_span, r0, t_eval=t_eval, method=method, rtol=rtol, atol=atol, args=(mooring_type, n_mooring, Z0, mooring_params_calced, env_params, witt_params, mooring_params, pto_flag))
        else:
            t_current = t_start
            function_current = WITT_function # start with the non power take off function
            r_current = r0
            pto_flag = False
            solution_event_t = np.array([])
            solution_event_r = np.empty((len(r0), 0))
            event_timestamps = np.array([])
            while t_current < t_end:
                print(f'pto_flag set to {pto_flag}')
                # run the ODE solver
                sol = solve_ivp(fun=function_current, t_span=(t_current, t_end), y0=r_current, t_eval=t_eval, method=method, events=PTO_event, rtol=rtol, atol=atol, args=(mooring_type, n_mooring, Z0, mooring_params_calced, env_params, witt_params, mooring_params, pto_flag))
                
                # save solution
                solution_event_t = np.hstack((solution_event_t, sol.t))
                solution_event_r = np.hstack((solution_event_r, sol.y))

                if sol.t_events[0].size == 0:
                    # no more events occurr; we're done
                    break
                
                # Update time and state etc
                t_current = sol.t_events[0][0]
                event_timestamps = np.hstack((event_timestamps, t_current))
                print(f'PTO event detected at t={t_current:.2f}s...')
                r_current = sol.y_events[0][0]
                t_eval = np.arange(t_current, t_end, t_step)
                pto_flag = not pto_flag

                # Switch the ODE function
                function_current = nonlinear_2D_WITT_PTO if function_current == WITT_function else WITT_function
                print(f'Switching to {function_current.__name__}...')
            sol.t = solution_event_t
            sol.y = solution_event_r
    print('ODE solver finished...')
    
    # return results
    print(f'-------Finished simulation for {WITT_function.__name__} with {n_mooring} {mooring_type} mooring lines------')
    return sol, Z0

def equilibrium_depth(R, rho_w, g, total_weight, Fzm_0): # returns: Z0, eq_Fb, S_offset
    # equilibrium depth
    def f(Z0):
        return (R + Z0)**2 * (2*R - Z0) - 3/(rho_w*np.pi*g) * (total_weight + Fzm_0)
    initial_guess = 0 #0.2*R
    Z0 = np.ndarray.item(fsolve(f, initial_guess))
    print(f'Z0 solution found as: {Z0} m (positive downwards)')
    
    eq_Fb = 1/3 * rho_w * g * np.pi * (R + Z0)**2 * (2*R - Z0) * np.cos(0) # buoyancy force at equilibrium
    print(f'Buoyancy force at equilibrium (Z0) is: {eq_Fb} N, weight of WITT and ballast is: {total_weight} N, mooring tension is: {Fzm_0} N')
    print(f'Weight + mooring tension is: {total_weight + Fzm_0} N')
    
    if abs(eq_Fb - total_weight - Fzm_0) > 1e-6:
        print('WARNING: Buoyancy force at equilibrium does not match the weight of the WITT and ballast plus mooring tension!')
        sys.exit(1) # stop code from running
    else:
        print('Buoyancy force at equilibrium matches the weight of the WITT and ballast plus mooring tension.')
    
    S_offset = np.pi * (R**2 - Z0**2) # area at waterline in equilibrium
    print('S_offset is: ', S_offset, 'm^2')
    
    if np.abs(Z0) > R:
        print(f'Error: Z0 = {Z0}m is outside of buoy Radius {R}m')
        sys.exit(1)
    
    return Z0, eq_Fb, S_offset

def check_initial_mooring(Fz, Fzm_0, max_mooring_Fz):
    if np.abs(np.abs(Fz) - Fzm_0) < Fzm_0*0.05: # compare with tolerance
        print(f'Initial vertical mooring force ({np.abs(Fz)}N) matches that from T0 ({Fzm_0}N), with tolerance {Fzm_0*0.05}N')
    else:
        print(f'WARNING: Initial vertical mooring force ({np.abs(Fz)}N) does not match that from T0 ({Fzm_0}N), with tolerance {Fzm_0*0.05}N')
        sys.exit(1)
    
    # check vertical mooring force is not too high
    if np.abs(Fz) > max_mooring_Fz:
        print('WARNING: vertical mooring force is too high! Buoy will sink!')
        print(f'Vertical mooring force = {np.abs(Fz)} N, max vertical mooring force = {max_mooring_Fz} N')
        sys.exit(1) # stop code from running
    else:
        print('Vertical mooring force is acceptable.')
        print(f'Vertical mooring force = {np.abs(Fz)} N, max vertical mooring force = {max_mooring_Fz} N')
    return

# function for nonlinear 2D (planar) WITT device
def nonlinear_2D_WITT(t, r, mooring_type, n_mooring, Z0, mooring_params_calced, env_params, witt_params, mooring_params, pto_flag):
    """nonlinear 2D ODE function for a WITT device in a spherical buoy

    Args:
        t (float): time
        r (list): state vector [x, dx/dt, z, dz/dt, theta, dtheta/dt]

    Returns:
        list: derivatives [dx/dt, d2x/dt2, dz/dt, d2z/dt2, dtheta/dt, d2theta/dt2]
    """
    x, dxdt, z, dzdt, theta, dthetadt, phi, dphidt = r
    
    # read in parameters# read in parameters from dictionaries
    g, rho_w, gamma_z, gamma_x, gamma_phi, a, v, phis, k_w, lambdas, n_waves = get_values_env(env_params)
    witt_size, m, m_s, m_pto, m_b, l, D, R, S, zeta0, c0, zeta, c, dtheta_PTO, thickness, M, J, total_weight, Ca, Cd = get_values_witt(witt_params)
    h, alpha_m, X0, T0z, T0x, lambda_m0, lambda_m, slack_allowed, Kx, Kz, Kphi, Kxphi, Kzphi = get_values_mooring(mooring_params) # args: h, alpha_m, X0, T0z, T0x, lambda_m0, lambda_m, slack_allowed, Kx, Kz, Kphi, Kxphi, Kzphi
    
    # wave forcing
    A, sigma_ave = wave_forcing_cairns(t, x, R, a, v, phis, lambdas, n_waves)  # wave forcing from Cairns et al approach
    
    # Buoyancy force - constrain h_s between 0 and D then calculate buoyancy force, centre of buoyancy, and moment - TODO: make this more robust? smaller wavelengths?
    h_s = R + (Z0 + A - z)*np.cos(sigma_ave) #*np.cos(theta)
    F_b, Moment_b = buoyancy_force(h_s, rho_w, g, R, D, sigma_ave, z, A, Z0) # NOTE: This previously slowed down code so change back to explicit code here if that happens
    
    # Horizontal hydrostatic force
    F_hydro, M_hydro, x_m = horizontal_hydrostatic_force(x, z, h_s, R, sigma_ave, rho_w, g)
    
    # Drag force
    F_drag_x, F_drag_z, M_drag_x, M_drag_phi, M_drag_z = drag_forces(x, z, Z0, A, R, sigma_ave, h_s, dxdt, dzdt, dphidt, rho_w, Cd, x_m) # returns: F_drag_x, F_drag_z, M_drag_x, M_drag_phi
    
    # mooring forces
    if n_mooring == 0:
        Fxm, Fzm, Fphim = 0, 0, 0
    else:
        if mooring_type == 'simplest_mooring':
            Fxm, Fzm, Fphim = simplest_mooring(t, x, z, phi, T0z, T0x, Kx, Kz, Kphi, Kxphi, Kzphi, n_mooring) #x, z, phi, T0z, T0x, Kx, Kz, Kphi, Kxphi, Kzphi, n_mooring
        elif mooring_type == 'simple_taut':
            Fxm, Fzm, Fphim = simple_taut_mooring(t, x, z, phi, R, n_mooring, Z0, alpha_m, h, lambda_m0, mooring_params_calced)
        elif mooring_type == 'piecewise':
            Fxm, Fzm, Fphim = piecewise_mooring(x, z, phi, R, n_mooring, Z0, alpha_m, h, lambda_m0, lambda_m, mooring_params_calced)
    
    # non-conservative forces
    Fx = Fxm + F_hydro + F_drag_x
    Fg = (M+m)*g
    Fz = Fzm + F_drag_z
    Ftheta = -c0*(dthetadt)
    Fphi = Fphim + M_hydro + M_drag_x + M_drag_phi + M_drag_z
    
    # print(f'sigma {sigma_ave}, mooring {Fphim}, hydro {M_hydro}, phi {phi}, dphidt {dphidt}, drag {M_drag_x+M_drag_phi+M_drag_z}')
    # print(f'Total moment {Fphi+Moment_b}, M buoy {Moment_b}, Fphi {Fphi}')
    
    # power take-off
    if not event_flag and pto_calc == True:
        if np.abs(dthetadt) > dtheta_PTO:
            epsilon = 1  # smoothing factor, smaller is sharper, aim for 0.05*dtheta_PTO or smaller
            activation = 0.5 * (1 + np.tanh((np.abs(dthetadt) - dtheta_PTO) / epsilon))
            c_eff = c0 + (c - c0) * activation
            Ftheta = -c_eff * (dthetadt)
            # Ftheta = -c*dthetadt  # PTO force in theta direction
        else:
            Ftheta = -c0*(dthetadt)
    else:
        Ftheta = -c0*(dthetadt)
    
    # equations of motion
    dxdt2 = Fx + m*l*(dthetadt+dphidt)**2*np.sin(theta+phi)
    dzdt2 = Fz - m*l*(dthetadt+dphidt)**2*np.cos(theta+phi) - Fg + F_b
    dthetadt2 = Ftheta - m*g*l*np.sin(theta+phi)
    dphidt2 = Fphi + Moment_b - m*g*l*np.sin(theta+phi) - R*m_b*g*np.sin(phi)
    
    # print(f'z acc {dzdt2}, z vel {dzdt}, Fzm {Fzm}, Fz drag {F_drag_z}, Fg {-Fg}, F_b {F_b}')
    # print(f'Fxm {Fxm}, F drag x {F_drag_x}, dxdt {dxdt}, x acc {dxdt2}, F hydro {F_hydro}. sigma {sigma_ave}')

    # dynamic mass matrix (dependent on theta)
    if h_s-R <= -R:
        Cm = 1 # not submerged so no added mass
    else:
        Cm = 1 + Ca # mass coefficient, i.e. normal mass + added mass
    Mmat = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, Cm*(M + m), 0, 0, 0, m*l*np.cos(theta+phi), 0, m*l*np.cos(theta+phi)],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, Cm*(M + m), 0, m*l*np.sin(theta+phi), 0, m*l*np.sin(theta+phi)],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, m*l*np.cos(theta+phi), 0, m*l*np.sin(theta+phi), 0, m*l**2, 0, m*l**2],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, m*l*np.cos(theta+phi), 0, m*l*np.sin(theta+phi), 0, m*l**2, 0, J+m*l**2]
    ])
    
    # f(r,t) vector
    fvec = np.array([
        dxdt,
        dxdt2,
        dzdt,
        dzdt2,
        dthetadt,
        dthetadt2,
        dphidt,
        dphidt2
    ])
    
    # solve for dr/dt
    rdot = np.linalg.solve(Mmat, fvec)
    return rdot

def nonlinear_2D_WITT_PTO(t, r, mooring_type, n_mooring, Z0, mooring_params_calced, env_params, witt_params, mooring_params, pto_flag):
    """nonlinear 2D ODE function for a WITT device in a spherical buoy

    Args:
        t (float): time
        r (list): state vector [x, dx/dt, z, dz/dt, theta, dtheta/dt]

    Returns:
        list: derivatives [dx/dt, d2x/dt2, dz/dt, d2z/dt2, dtheta/dt, d2theta/dt2]
    """
    x, dxdt, z, dzdt, theta, dthetadt, phi, dphidt = r
    
    # read in parameters# read in parameters from dictionaries
    g, rho_w, gamma_z, gamma_x, gamma_phi, a, v, phis, k_w, lambdas, n_waves = get_values_env(env_params)
    witt_size, m, m_s, m_pto, m_b, l, D, R, S, zeta0, c0, zeta, c, dtheta_PTO, thickness, M, J, total_weight, Ca, Cd = get_values_witt(witt_params)
    h, alpha_m, X0, T0z, T0x, lambda_m0, lambda_m, slack_allowed, Kx, Kz, Kphi, Kxphi, Kzphi = get_values_mooring(mooring_params) # args: h, alpha_m, X0, T0z, T0x, lambda_m0, lambda_m, slack_allowed, Kx, Kz, Kphi, Kxphi, Kzphi
    
    # wave forcing
    A, sigma_ave = wave_forcing_cairns(t, x, R, a, v, phis, lambdas, n_waves)  # wave forcing from Cairns et al approach
    
    # Buoyancy force - constrain h_s between 0 and D then calculate buoyancy force, centre of buoyancy, and moment - TODO: make this more robust? smaller wavelengths?
    h_s = R + (Z0 + A - z)*np.cos(sigma_ave) #*np.cos(theta)
    F_b, Moment_b = buoyancy_force(h_s, rho_w, g, R, D, sigma_ave, z, A, Z0) # NOTE: This previously slowed down code so change back to explicit code here if that happens
    
    # Horizontal hydrostatic force
    F_hydro, M_hydro, x_m = horizontal_hydrostatic_force(x, z, h_s, R, sigma_ave, rho_w, g)
    
    # Drag force
    F_drag_x, F_drag_z, M_drag_x, M_drag_phi, M_drag_z = drag_forces(x, z, Z0, A, R, sigma_ave, h_s, dxdt, dzdt, dphidt, rho_w, Cd, x_m) # returns: F_drag_x, F_drag_z, M_drag_x, M_drag_phi
    
    # mooring forces
    if n_mooring == 0:
        Fxm, Fzm, Fphim = 0, 0, 0
    else:
        if mooring_type == 'simplest_mooring':
            Fxm, Fzm, Fphim = simplest_mooring(t, x, z, phi, T0z, T0x, Kx, Kz, Kphi, Kxphi, Kzphi, n_mooring) #x, z, phi, T0z, T0x, Kx, Kz, Kphi, Kxphi, Kzphi, n_mooring
        elif mooring_type == 'simple_taut':
            Fxm, Fzm, Fphim = simple_taut_mooring(t, x, z, phi, R, n_mooring, Z0, alpha_m, h, lambda_m0, mooring_params_calced)
        elif mooring_type == 'piecewise':
            Fxm, Fzm, Fphim = piecewise_mooring(x, z, phi, R, n_mooring, Z0, alpha_m, h, lambda_m0, lambda_m, mooring_params_calced)
    
    # non-conservative forces
    Fx = Fxm + F_hydro + F_drag_x
    Fg = (M+m)*g
    Fz = Fzm + F_drag_z
    Ftheta = -c0*(dthetadt)
    Fphi = Fphim + M_hydro + M_drag_x + M_drag_phi + M_drag_z
    
    # power take-off
    if np.abs(dthetadt) > dtheta_PTO:
        # epsilon = 0.1*dtheta_PTO**2 #0.04*dethat_PTO**3 #0.5  # smoothing factor, smaller is sharper, aim for 0.05*dtheta_PTO or smaller
        # # using dtheta_PTO**2 because when threshold is low the curve can be sharper, yet for higher thresholds the curve NEEDS to be smoother
        # activation = 0.5 * (1 + np.tanh((np.abs(dthetadt) - dtheta_PTO) / epsilon))
        # c_eff = c0 + (c - c0) * activation
        # Ftheta = -c_eff * dthetadt
        Ftheta = -c*(dthetadt)  # PTO force in theta direction
    
    # equations of motion
    dxdt2 = Fx + m*l*(dthetadt+dphidt)**2*np.sin(theta+phi)
    dzdt2 = Fz - m*l*(dthetadt+dphidt)**2*np.cos(theta+phi) - Fg + F_b
    dthetadt2 = Ftheta - m*g*l*np.sin(theta+phi)
    dphidt2 = Fphi + Moment_b - m*g*l*np.sin(theta+phi) - R*m_b*g*np.sin(phi)
    
    # dynamic mass matrix (dependent on theta)
    if h_s-R <= -R:
        Cm = 1 # not submerged so no added mass
    else:
        Cm = 1 + Ca # mass coefficient, i.e. normal mass + added mass
    Mmat = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, Cm*(M + m), 0, 0, 0, m*l*np.cos(theta+phi), 0, m*l*np.cos(theta+phi)],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, Cm*(M + m), 0, m*l*np.sin(theta+phi), 0, m*l*np.sin(theta+phi)],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, m*l*np.cos(theta+phi), 0, m*l*np.sin(theta+phi), 0, m*l**2, 0, m*l**2],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, m*l*np.cos(theta+phi), 0, m*l*np.sin(theta+phi), 0, m*l**2, 0, J+m*l**2]
    ])
    
    # f(r,t) vector
    fvec = np.array([
        dxdt,
        dxdt2,
        dzdt,
        dzdt2,
        dthetadt,
        dthetadt2,
        dphidt,
        dphidt2
    ])
    
    # solve for dr/dt
    rdot = np.linalg.solve(Mmat, fvec)
    return rdot

def PTO_event(t, r, *args): #args=(mooring_type, n_mooring, Z0, mooring_params_calced, env_params, witt_params, mooring_params, pto_flag)
    dthetadt = r[5]
    _,_,_,_,_,witt_params,_,pto_flag = args
    witt_size, m, m_s, m_pto, m_b, l, D, R, S, zeta0, c0, zeta, c, dtheta_PTO, thickness, M, J, total_weight, Ca, Cd = get_values_witt(witt_params)
    HYSTERESIS_MARGIN = dtheta_PTO*0.1
    if pto_flag:
        return np.abs(dthetadt) - (dtheta_PTO - HYSTERESIS_MARGIN)
    else:
        return np.abs(dthetadt) - (dtheta_PTO + HYSTERESIS_MARGIN)
PTO_event.direction = 0   # detect both upward and downward crossings
PTO_event.terminal = True # terminate integration and swap to other model

# function to calculate wave forcing
def wave_forcing_cairns(t, x, R, a, v, phis, lambdas, n_waves):
    # wave height
    # A_temp = np.zeros(n_waves)
    # for i in range(n_waves):
    #     A_temp[i] = a[i] * np.sin(2*np.pi * (v[i]*t - x/lambdas[i]) + phis[i])
    # A = np.sum(A_temp)
    
    A_list = []
    for x_temp in np.array([x-R, x, x+R]):
        A_temp = np.zeros(n_waves)
        for i in range(n_waves):
            A_temp[i] = a[i] * np.sin(2*np.pi * (v[i]*t - x/lambdas[i]) + phis[i])
            # A_temp[i] = float(a[i] * np.sin(2*np.pi * (float(v[i])*t - x/float(lambdas[i])) + float(phis[i])))
        A_list.append(np.sum(A_temp))
    A_ave = np.mean(A_list)
    
    # surface angle
    sigma_list = []
    for x_temp in np.array([x-R, x, x+R]):
        sigma_temp = np.zeros(n_waves)
        for i in range(n_waves):
            sigma_temp[i] = 2*np.pi*a[i]/lambdas[i] * np.cos(2*np.pi*(v[i]*t - x_temp/lambdas[i]) + phis[i])
        sigma_list.append(-np.arctan(np.sum(sigma_temp)))
    sigma_ave = np.mean(sigma_list)
    
    return A_ave, sigma_ave

def buoyancy_force(h_s, rho_w, g, R, D, sigma_ave, z, A, Z0):
    if h_s-R <= -R: #h_s <= 0
        # print(f'Not submerged with h_s {h_s}, z {z}, A {A}, sigma {sigma_ave}, Z0 {Z0}')
        h_s = 0
        F_b = 0
        Moment_b = 0
    elif h_s-R >= R: #h_s >= D:
        h_s = D
        F_b = 4/3 * rho_w * g * np.pi * R**3#* np.cos(sigma_ave) # buoyancy force at full submersion
        CoB = 0
        Moment_b = 0
        # print('Fully submerged!!')
    else:
        CoB = 3/4 * (2*R - h_s)**2 / (3*R - h_s)
        F_b = 1/3 * rho_w * g * np.pi * h_s**2 * (3*R - h_s)# * np.cos(sigma_ave)
        Moment_b = F_b * np.sin(sigma_ave) * CoB
    
    # print(sigma_ave, Moment_b)
    # print(h_s, F_b)
    return F_b, Moment_b

def horizontal_hydrostatic_force(x, z, h_s, R, sigma_ave, rho_w, g):
    if h_s-R <= -R:
        F_hydro, M_hydro = 0, 0
        x_m = x
        # print('not submerged')
    elif h_s-R >= R:
        F_hydro, M_hydro = 0, 0
        x_m = x
        # print('fully submerged')
    else:
        # find coordinates at middle of buoy cross section made by water surface at amplitude A and angle sigma
        x_m = x - np.abs((h_s-R)*np.sin(sigma_ave))
        z_m = z + (h_s-R)*np.cos(sigma_ave)
        if (h_s-R)>R or (h_s-R)<-R:
            print(f'Error: submerged height h_s, {h_s-R}m, outside of buoy diameter {2*R}m, yet calculating hydrostatic force')
            sys.exit(1)
        R_s = np.sqrt(R**2 - (h_s-R)**2) # radius of cross section
        # find x intersection coordinates of waves and buoy
        x_l = x_m - R_s*np.cos(sigma_ave)
        z_l = z_m - R_s*np.sin(sigma_ave)
        x_r = x_m + R_s*np.cos(sigma_ave)
        z_r = z_m + R_s*np.sin(sigma_ave)
        # find segment areas and centroids
        d_l = np.clip(z_l - z, -R, R)
        d_r = np.clip(z_r - z, -R, R)
        delta_l = 2*np.arccos(d_l/R)
        delta_r = 2*np.arccos(d_r/R)
        Area_seg_l = R**2/2 * (delta_l - np.sin(delta_l))
        Area_seg_r = R**2/2 * (delta_r - np.sin(delta_r))
        # find submerged ares
        Area_left = np.pi*R**2 - Area_seg_l
        Area_right = np.pi*R**2 - Area_seg_r
        # find submerged centroids
        # if Area_left<1e-3 or Area_right<1e-3:
        #     print(f'Warning, small submerged areas: left {Area_left}m2, right {Area_right}m2')
        #     print(f'Deltas: left {delta_l}, right {delta_r}')
        angle_tol = 1e-2
        area_tol = 1e-3
        if abs(delta_l) < angle_tol:
            z_seg_l = 0 # for small angles, segment above water is effectively non-existant so set to 0
            z_cl = z # therefore submerged area centroid is effectively buoy centre z
            # print('delta_l smaller than tolerance')
        elif abs(2*np.pi - delta_l) < angle_tol:
            z_seg_l = z # for large angles, segment above water is effectively the whole thing so set to circle centroid z
            z_cl = z-R # therefore submerged centroid is effectively at z-R, and submerged area = 0
            Area_left = 0
            # print('delta_l close to 2pi')
        else:
            z_seg_l = z + (4*R*np.sin(delta_l/2)**3) / (3*(delta_l-np.sin(delta_l)))
            if Area_left < area_tol:
                z_cl = z-R
                # print('Area_left smaller than tolerance')
            else:
                z_cl = (z*np.pi*R**2 - Area_seg_l*z_seg_l)/Area_left #,np.clip( ,-R, R)
                # print('z_cl calculated normally')
        
        if abs(delta_r) < angle_tol:
            z_seg_r  = 0
            z_cr = z
            # print('delta_r smaller than tolerance')
        elif abs(2*np.pi - delta_r) < angle_tol:
            z_seg_r = z
            z_cr = z-R
            Area_right = 0
            # print('delta_r close to 2pi')
        else:
            z_seg_r = z + (4*R*np.sin(delta_r/2)**3) / (3*(delta_r-np.sin(delta_r)))
            if Area_right < area_tol:
                z_cr = z-R
                # print('Area_right smaller than tolerance')
            else:
                z_cr = (z*np.pi*R**2 - Area_seg_r*z_seg_r)/Area_right
                # print('z_cr calculated normally')
        # print(f'Submerged areas set to: left {Area_left}m2, right {Area_right}m2')
        # print(f'And centroids (from buoy z) are: left {z_cl-z}, right {z_cr-z}')
        # print(f'So submerged depths of centroids are: left {z_l-z_cl}, right {z_r-z_cr}')
        
        # find forces
        F_hydro_left = rho_w*g*(z_l-z_cl)*Area_left
        F_hydro_right = rho_w*g*(z_r-z_cr)*Area_right
        
        # find moments
        M_hydro_left = F_hydro_left*(z-z_cl)
        M_hydro_right = F_hydro_right*(z-z_cr)
        # print(f'Forces and Moments: F left {F_hydro_left}, F right {F_hydro_right}, M left {M_hydro_left}, M right {M_hydro_right}')
        
        # check if intersection points cross vertical centreline of buoy, if so projected area is that of the full submerged buoy
        if x_l > x:
            F_hydro_left = rho_w*g*R*np.pi*R**2
            M_hydro_left = 0
        if x_r < x:
            F_hydro_right = rho_w*g*R*np.pi*R**2
            M_hydro_right = 0
        
        # find total forces and moments
        F_hydro = F_hydro_left - F_hydro_right
        M_hydro = M_hydro_left - M_hydro_right

        # check z_cl coordinates are within buoy radius, as this caused issues previously 
        if np.abs(z_cl-z)>R+1e-3 or np.abs(z_cr-z)>R+1e-3:
            print("Warning, this message shouldn't print")
            print(z_cl-z, z_cr-z)
        
        # print(F_hydro, M_hydro)
        # set to 0 for testing
        # F_hydro, M_hydro = 0, 0
    
    # print(sigma_ave, M_hydro)
    return F_hydro, M_hydro, x_m

def drag_forces(x, z, Z0, A, R, sigma_ave, h_s, dxdt, dzdt, dphidt, rho_w, Cd, x_m): # returns: F_drag_x, F_drag_z, M_drag_x, M_drag_phi
    # x ref area calcs
    arg = np.clip((Z0-z+A)/R, -1, 1)
    delta_drag = 2*np.pi - 2*np.arccos(arg) # find angle of segment above water, then subtract from 2pi for angle of segment below
    ref_area_x = R**2/2 * (delta_drag - np.sin(delta_drag)) # average vertical submerged area
    ref_area_x = max(ref_area_x, 0.0)
    if np.abs(delta_drag) < 1e-7:
        z_drag = R
    elif np.abs(2*np.pi - delta_drag) < 1e-7:
        z_drag = 0
    else:
        z_drag = (4*R*np.sin(delta_drag/2)**3) / (3*(delta_drag-np.sin(delta_drag))) # centroid of drag reference area (for force application/moment)
    
    # z ref area calcs
    ref_area_z = np.cos(sigma_ave)*np.pi*(R**2 - (h_s-R)**2) # projected area (in xy plane) between 2 intersection points
    if h_s-R <= -R:
        ref_area_z = 0 #np.pi*R**2 * 1.225/rho_w
    elif h_s-R >= 0:
        ref_area_z = np.pi*R**2
    else:
        ref_area_z = max(ref_area_z, 0.0)
    # ref_area_z = max(ref_area_z, 0.0)
    # print(ref_area_z)
    x_drag = x_m-x
    
    # drag calcs
    dxdt_rel = dxdt # -TODO: update these to actually be relative velocity of water compared to buoy??
    dzdt_rel = dzdt
    dphidt_rel = dphidt
    Cdt = 0.1 # tangential drag coefficient
    # print(f'Drag terms: arg {arg}, {delta_drag}, ref_area {ref_area}, z_drag {z_drag}')
    F_drag_x = -0.5 * rho_w * Cd * ref_area_x * (dxdt_rel) * np.abs(dxdt_rel)
    F_drag_z = -0.5 * rho_w * Cd * ref_area_z * (dzdt_rel) * np.abs(dzdt_rel)
    M_drag_x = F_drag_x*z_drag
    M_drag_z = F_drag_z*x_drag
    M_drag_phi = -0.5 * rho_w * Cdt * ref_area_x * (dphidt_rel) * np.abs(dphidt_rel) * R
    # print(f'Drag forces: Fx {F_drag_x}, Mx {M_drag_x}')
    # print(M_drag_z, F_drag_z, x_drag)
    # print(M_drag_phi)
    
    # error checking
    drag_error_array = np.array([arg, delta_drag, ref_area_x, z_drag, F_drag_x, M_drag_x])
    if not np.all(np.isfinite(drag_error_array)):
        print("Error: array contains NaN or Inf")
        print("Values:", drag_error_array)

    return F_drag_x, F_drag_z, M_drag_x, M_drag_phi, M_drag_z

# define function for the simplest mooring
def simplest_mooring(t, x, z, phi, T0z, T0x, Kx, Kz, Kphi, Kxphi, Kzphi, n_mooring):
    if n_mooring == 1:
        Fx = -T0x - Kx*x - Kxphi*phi
        Fz = -T0z - Kz*z + Kzphi*phi
        Mphi = -Kphi*phi - Kxphi*x + Kzphi*z
        if Fx > 0:
            Fx = 0
            print(f'Warning: Fxm = 0 at t = {t}')
        if Fz > 0:
            Fz = 0
            print(f'Warning: Fzm = 0 at t = {t}')
        
        # if x > 0:
        #     Fx = -Kx*x
        #     Mphi = -Kxphi*x
        # else:
        #     Fx = 0
        #     Mphi = 0
        
        # if phi > 0:
        #     Fx = Fx - Kxphi*phi
        # else:
        #     Fx = Fx
        
        # Fz = -T0z - Kz*z
        # if Fz > 0:
        #     Fz = 0
        #     print('Warning: Fzm = 0')
        # Mphi = Mphi - Kphi*phi
    elif n_mooring == 2:
        Fx = -Kx*x - Kxphi*phi
        Fz = -T0z - Kz*z
        if Fz > 0:
            Fz = 0
            print(f'Warning: Fzm = 0, z={z} at t = {t}')
        Mphi = -Kphi*phi - Kxphi*x
    
    return Fx, Fz, Mphi

# define function for simple_taut mooring forces
def simple_taut_mooring(t, x, z, phi, R, n_mooring, Z0, alpha_m, h, lambda_m0, mooring_params_calced):
    
    # read in params
    h1, h2, L0_true = mooring_params_calced
    
    # change in attachment point coordinates
    x_a1 = R*np.sin(alpha_m)-R*np.sin(alpha_m-phi) #*np.cos(np.pi/2 - alpha_m + phi)
    dx_a1 = x + x_a1
    x_a2 = R*np.sin(phi+alpha_m)-R*np.sin(alpha_m)
    dx_a2 = x + x_a2
    z_a1 = R*np.cos(alpha_m) - R*np.cos(alpha_m-phi)
    dz_a1 = z + z_a1
    z_a2 = R*np.cos(alpha_m) - R*np.cos(alpha_m+phi)
    dz_a2 = z + z_a2
    
    # angles of mooring lines
    X1 = np.arctan((h2+dx_a1)/(h1+dz_a1)) #np.arctan((h2 + x_a1)/(h-Z0-z_a1))
    X2 = np.arctan((h2-dx_a2)/(h1+dz_a2)) #np.arctan((h2 - x_a2)/(h-Z0-z_a2))
    
    # calculate true extension
    delta_L1 = np.sqrt((h2 + dx_a1)**2 + (h1+dz_a1)**2) - L0_true  # effective change in length of mooring line for buoy 1
    delta_L2 = np.sqrt((h2 - dx_a2)**2 + (h1+dz_a2)**2) - L0_true  # effective change in length of mooring line for buoy 2
    if delta_L1 < 0:
        print(f'Warning: mooring line delta_L1 < 0, ({delta_L1} at t = {t}s)')
    if delta_L2 < 0:
        print(f'Warning: mooring line delta_L2 < 0, ({delta_L2} at t = {t}s)')
    
    # ensure greater than 0
    delta_L1 = max(delta_L1, 0)
    delta_L2 = max(delta_L2, 0)
    
    # calulate spring force
    F1 = lambda_m0 * delta_L1  # force in mooring line for buoy 1
    F2 = lambda_m0 * delta_L2  # force in mooring line for buoy 2
    
    # calculate and forces and moments
    Fx1 = - F1 * np.sin(X1)
    Fx2 = F2 * np.sin(X2)
    Fz1 = - F1 * np.cos(X1)
    Fz2 = - F2 * np.cos(X2)
    if n_mooring == 1:
        Fx = Fx1
        Fz = Fz1
        Mphi = Fx1*R*np.cos(alpha_m - phi) - Fz1*R*np.sin(alpha_m - phi)
    elif n_mooring == 2:
        Fx = Fx1 + Fx2
        Fz = Fz1 + Fz2
        Mphi = Fx1*R*np.cos(alpha_m - phi) + Fx2*R*np.cos(alpha_m + phi) - Fz1*R*np.sin(alpha_m - phi) + Fz2*R*np.sin(alpha_m + phi)
    
    return Fx, Fz, Mphi

# define function for piecewise mooring forces
def piecewise_mooring(x, z, phi, R, n_mooring, Z0, alpha_m, h, lambda_m0, lambda_m, mooring_params_calced):
    """Piecewise mooring force function

    Args:
        x (float): x position of buoy
        z (float): z position of buoy
        phi (float): angle of buoy

    Returns:
        tuple: mooring forces (Fx, Fz, Mphi)
    """
    # read in mooring parameters
    h1, h2, L0_true, slack_limit = mooring_params_calced
    
    # change in attachment point coordinates
    x_a1 = R*np.sin(alpha_m)-R*np.sin(alpha_m-phi) #*np.cos(np.pi/2 - alpha_m + phi)
    dx_a1 = x + x_a1
    x_a2 = R*np.sin(phi+alpha_m)-R*np.sin(alpha_m)
    dx_a2 = x + x_a2
    z_a1 = R*np.cos(alpha_m) - R*np.cos(alpha_m-phi)
    dz_a1 = z + z_a1
    z_a2 = R*np.cos(alpha_m) - R*np.cos(alpha_m+phi)
    dz_a2 = z + z_a2
    
    # angles of mooring lines
    X1 = np.arctan((h2+dx_a1)/(h1+dz_a1)) #np.arctan((h2 + x_a1)/(h-Z0-z_a1))
    X2 = np.arctan((h2-dx_a2)/(h1+dz_a2)) #np.arctan((h2 - x_a2)/(h-Z0-z_a2))
    
    # calculate true extension
    delta_L1 = np.sqrt((h2 + dx_a1)**2 + (h1+dz_a1)**2) - L0_true  # effective change in length of mooring line for buoy 1
    delta_L2 = np.sqrt((h2 - dx_a2)**2 + (h1+dz_a2)**2) - L0_true  # effective change in length of mooring line for buoy 2
    
    # ensure greater than 0
    delta_L1 = max(delta_L1, 0)
    delta_L2 = max(delta_L2, 0)
    
    # check if slack or taut
    if  delta_L1 < slack_limit:
        # mooring line is slack, use slack stiffness
        F1 = lambda_m0 * delta_L1
    elif delta_L1 >= slack_limit:
        # mooring line is taut, use taut stiffness
        F1 = lambda_m0 * slack_limit + lambda_m * (delta_L1-slack_limit)
    
    if delta_L2 < slack_limit:
        # mooring line is slack, use slack stiffness
        F2 = lambda_m0 * delta_L2
    elif delta_L2 >= slack_limit:
        # mooring line is taut, use taut stiffness
        F2 = lambda_m0 * slack_limit + lambda_m * (delta_L2-slack_limit)
    
    # calculate and forces and moments
    Fx1 = - F1 * np.sin(X1)
    Fx2 = F2 * np.sin(X2)
    Fz1 = - F1 * np.cos(X1)
    Fz2 = - F2 * np.cos(X2)
    if n_mooring == 1:
        Fx = Fx1
        Fz = Fz1
        Mphi = Fx1*R*np.cos(alpha_m - phi) - Fz1*R*np.sin(alpha_m - phi)
    elif n_mooring == 2:
        Fx = Fx1 + Fx2
        Fz = Fz1 + Fz2
        Mphi = Fx1*R*np.cos(alpha_m - phi) + Fx2*R*np.cos(alpha_m + phi) - Fz1*R*np.sin(alpha_m - phi) + Fz2*R*np.sin(alpha_m + phi)
    
    return Fx, Fz, Mphi

def update_params_env(env_params, **kwargs):
    for key, value in kwargs.items():
        if key in env_params:
            if key in ['a', 'v', 'phis']:
                env_params[key] = np.array([value])
                print(f"Updated '{key}' to {value} as an array")
            else:
                env_params[key] = value
                print(f"Updated '{key}' to {value}")
        else:
            print(f"Warning: '{key}' not found in environmental parameters.")
    
    print('Updating derived environemental parameters')
    env_params = update_derived_params_env(env_params)
    
    return env_params

def update_derived_params_env(env_params):
    g = env_params['g']
    v = env_params['v']
    h = mooring_params['h']
    k_w = find_kw(v, h) #(2 * np.pi * v) ** 2 / g #np.sqrt((2 * np.pi * v) ** 2 / g)
    env_params['k_w'] = k_w
    env_params['lambdas'] = 2 * np.pi / k_w
    env_params['n_waves'] = len(env_params['a'])
    print(f'Wavenumbers are: {env_params['k_w']}')
    print(f"Wavelengths are: {env_params['lambdas']}")
    return env_params   

def find_kw(v, h, g = 9.81, tol=1e-12, maxiter=50):
    # omega = 2*np.pi*v # convert frequency to rad
    # k0 = omega**2/g # wave number in deep water
    # x0 = np.sqrt(k0*h) # initial guess of kh
    
    # # iterate solution
    # for i in range(maxiter):
    #     arg = (x0/np.cosh(x0))**2
    #     kh = x0*((k0*h + (arg) )/(x0*np.tanh(x0) + arg))
    #     if abs(x0-kh) < tol:
    #         # print(f'iterations used: {i}')
    #         return kh/h
    #     x0 = kh
    
    # return kh/h

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

def get_values_env(env_params): # args: g, rho_w, gamma_z, gamma_x, gamma_phi, a, v, phis, k_w, lambdas, n_waves
    return (
        env_params['g'], env_params['rho_w'],
        env_params['gamma_z'], env_params['gamma_x'], env_params['gamma_phi'],
        env_params['a'], env_params['v'], env_params['phis'],
        env_params['k_w'], env_params['lambdas'], env_params['n_waves']
    )

def update_params_witt(witt_params, **kwargs):
    for key, value in kwargs.items():
        if key in witt_params:
            witt_params[key] = value
            print(f"Updated '{key}' to {value}")
        else:
            print(f"Warning: '{key}' not found in WITT parameters.")
    
    print('Updating derived WITT parameters')
    witt_params = update_derived_params_witt(witt_params)
    return witt_params

def update_derived_params_witt(witt_params):
    m_s = witt_params['m_s']
    m_pto = witt_params['m_pto']
    m_b = witt_params['m_b']
    m = witt_params['m']
    l = witt_params['l']
    D = witt_params['D']
    R = D / 2
    thickness = witt_params['thickness']
    zeta0 = witt_params['zeta0']
    zeta = witt_params['zeta']
    g = env_params['g']
    
    M = m_s + m_pto + m_b
    S = np.pi * R**2
    total_weight = (M+m)*g
    c0 = 2*m*l*np.sqrt(g*l)*zeta0
    c = 2*m*l*np.sqrt(g*l)*zeta

    # Moment of inertia J (with optional shell thickness)
    if thickness == 0:
        J = R**2 * (m_b + (2/3) * m_s)
    elif thickness > 0:
        try:
            J = R**2 * (m_b + (2/5) * m_s * ((1 - (R - thickness)**5 / R**5) / (1 - (R - thickness)**3 / R**3)))
        except ZeroDivisionError:
            J = np.nan
            print("WARNING: Division by zero in inertia calc.")
    else:
        print("WARNING: negative thickness for spherical buoy")
        J = np.nan
    
    # Update derived values in the dictionary
    witt_params.update({
        'M': M,
        'R': R,
        'S': S,
        'J': J,
        'total_weight': total_weight,
        'c0': c0,
        'c': c
    })
    
    print(f'Derived WITT params updated to: M {witt_params['M']}, R {witt_params['R']}, S {witt_params['S']}, J {witt_params['J']}, total_weight {witt_params['total_weight']}, c0 {witt_params['c0']}, c {witt_params['c']}')
    return witt_params

def get_values_witt(witt_params): # args: witt_size, m, m_s, m_pto, m_b, l, D, R, S, zeta0, c0, zeta, c, dtheta_PTO, thickness, M, J, total_weight, Ca, Cd
    return (
        witt_params['witt_size'],
        witt_params['m'],         # total pendulum mass
        witt_params['m_s'],       # spherical buoy mass
        witt_params['m_pto'],     # PTO mass
        witt_params['m_b'],       # ballast mass
        witt_params['l'],         # pendulum length
        witt_params['D'],         # buoy diameter
        witt_params['R'],         # buoy radius
        witt_params['S'],         # buoy cross-sectional area
        witt_params['zeta0'],     # natural damping ratio
        witt_params['c0'],        # natural damping coefficient
        witt_params['zeta'],      # PTO damping ratio
        witt_params['c'],         # PTO damping coefficient
        witt_params['dtheta_PTO'],# dtheta threshold for PTO activation
        witt_params['thickness'], # buoy shell thickness
        witt_params['M'],         # total system mass
        witt_params['J'],         # moment of inertia
        witt_params['total_weight'], # total weight of WITT and ballast
        witt_params['Ca'], # added mass coefficient
        witt_params['Cd'] # drag coefficient
    )

def calc_nat_freqs(env_params, witt_params):
    # calculate buoy radius of gyration
    buoy_r_gyration = np.sqrt(witt_params['J']/witt_params['M']) # radius of gyration of buoy (m)

    # calculate natural frequencies
    f_nat_p = 1/(2*np.pi) * np.sqrt(env_params['g']/witt_params['l']) # natural frequency of pendulum (Hz)
    f_nat_b = 1/(2*np.pi) * np.sqrt(env_params['g']/witt_params['R']) # natural frequency of buoy (Hz)
    return buoy_r_gyration, f_nat_p, f_nat_b

def print_key_params(env_params, witt_params, f_nat_p, f_nat_b, buoy_r_gyration):
    # get values
    g, rho_w, gamma_z, gamma_x, gamma_phi, a, v, phis, k_w, lambdas, n_waves= get_values_env(env_params)
    witt_size, m, m_s, m_pto, m_b, l, D, R, S, zeta0, c0, zeta, c, dtheta_PTO, thickness, M, J, total_weight, Ca, Cd = get_values_witt(witt_params)
    print(f'WITT parameters: pendulum mass {m}kg, PTO mass {m_pto}kg, pendulum length {l}m')
    print(f'WITT pendulum natural frequency: {f_nat_p}Hz')
    print(f'Buoy parameters: diameter {D}m, radius {R}m, shell mass {m_s}kg, ballast mass {m_b}kg, total mass (buoy+WITT) {M+m}kg, radius of gyration {buoy_r_gyration}m')
    print(f'Buoy natural frequency: {f_nat_b}Hz')

def check_can_float(env_params, witt_params):
    # get values
    witt_size, m, m_s, m_pto, m_b, l, D, R, S, zeta0, c0, zeta, c, dtheta_PTO, thickness, M, J, total_weight, Ca, Cd = get_values_witt(witt_params)
    g, rho_w, gamma_z, gamma_x, gamma_phi, a, v, phis, k_w, lambdas, n_waves = get_values_env(env_params)  # Only need g and rho_w here

    # check not too heavy to float
    max_Fb = 4/3 * np.pi * R**3 * rho_w * g
    max_mooring_Fz = max_Fb - total_weight
    if total_weight > max_Fb:
        print('WARNING: WITT is too heavy to float! Max buoyancy force is not enough to support the weight of the WITT and ballast.')
        print(f'Weight of WITT and ballast = {total_weight} N, max buoyancy force = {max_Fb} N')
        sys.exit(1) # stop code from running
    else:
        print('WITT is light enough to float! Max buoyancy force is enough to support the weight of the WITT and ballast.')
        print(f'Weight of WITT and ballast = {total_weight} N, max buoyancy force = {max_Fb} N')
        print((f'Vertical mooring force must not exceed {max_mooring_Fz} N, otherwise WITT will be below the surface!'))
    
    return max_Fb, max_mooring_Fz

def update_params_mooring(mooring_params, **kwargs):
    for key, value in kwargs.items():
        if key in mooring_params:
            mooring_params[key] = value
            print(f"Updated '{key}' to {value}")
            if key == 'lambda_frac' or key == 'lambda_m0':
                mooring_params['lambda_m'] = mooring_params['lambda_m0'] * mooring_params['lambda_frac']
                print(f"Updated 'lambda_m' to {mooring_params['lambda_m']}")
        else:
            print(f"Warning: '{key}' not found in mooring parameters.")
    print('Mooring parameters updated.')
    return mooring_params

def get_values_mooring(mooring_params): # args: h, alpha_m, X0, T0z, T0x, lambda_m0, lambda_m, slack_allowed, Kx, Kz, Kphi, Kxphi, Kzphi
    return (
        mooring_params['h'], mooring_params['alpha_m'], mooring_params['X0'],
        mooring_params['T0z'], mooring_params['T0x'], mooring_params['lambda_m0'], mooring_params['lambda_m'], mooring_params['slack_allowed'],
        mooring_params['Kx'], mooring_params['Kz'], mooring_params['Kphi'], mooring_params['Kxphi'], mooring_params['Kzphi']
    )

def update_witt_size(witt_size, witt_params):
    if witt_size == 'small':
        witt_params.update({
                'witt_size': 'small',
                'm': 5.854,
                'm_s': 15,
                'm_pto': 2.61,
                'm_b': 15,
                'l': 0.1118,
                'D': 0.5,
                'zeta0': 0.05,
                'zeta': 0.2,
                'dtheta_PTO': 4.5, # threshold angular velocity for PTO
                'thickness': 0.02,
                'Ca': 0.5,
                'Cd': 0.5
            })
    elif witt_size == 'large':
        witt_params.update({
                'witt_size': 'large',
                'm': 110,
                'm_s': 30,
                'm_pto': 100,
                'm_b': 30,
                'l': 0.506,
                'D': 1.5,
                'zeta0': 0.05,
                'zeta': 0.2,
                'dtheta_PTO': 4.5, # threshold angular velocity for PTO
                'thickness': 0.02,
                'Ca': 0.5,
                'Cd': 0.5
            })
    else:
        raise ValueError("witt_size must be 'small' or 'large'")
    witt_params = update_derived_params_witt(witt_params)
    return witt_params

def wrap_angles(time, angle, threshold=np.pi):
    # Wrap angles to [-pi, pi]
    wrapped = (angle + np.pi) % (2 * np.pi) - np.pi
    
    # Identify discontinuities where the jump exceeds the threshold
    diffs = np.abs(np.diff(wrapped))
    discontinuities = np.where(diffs > threshold)[0]
    
    # Insert NaNs into both time and angle to break the plot
    time_wrapped = time.copy()
    angle_wrapped = wrapped.copy()
    for idx in reversed(discontinuities):  # reverse to preserve indices
        time_wrapped = np.insert(time_wrapped, idx + 1, np.nan)
        angle_wrapped = np.insert(angle_wrapped, idx + 1, np.nan)
    
    return time_wrapped, angle_wrapped

#%% define parameters
# initialise mooring parameters
mooring_params = {
    'h': 5,  # depth of water (m)
    'alpha_m': np.pi/2,  # angle between mooring point on hull and vertical (rad)
    'X0': np.pi/4,  # angle between hypotenuse of mooring line and vertical
    'T0z': 100,  # prestressed tension in z (N)
    'T0x': 100, # prestressed tension in x (N)
    'lambda_m0': 50,  # mooring stiffness (N/m)
    'lambda_frac': 10,
    'slack_allowed': 1,
    'Kx': 200,
    'Kz': 100,
    'Kphi': 50,
    'Kxphi': 0,
    'Kzphi': 50
}
mooring_params['lambda_m'] = mooring_params['lambda_m0']*mooring_params['lambda_frac'] # mooring stiffness when taut, 10* of slack stiffness

# initialise environmental parameters
env_params = { # define dictionary of parameters
    'g': 9.81,  # gravity
    'rho_w': 1025,  # water density
    'gamma_z': 0,
    'gamma_x': 0,
    'gamma_phi': 0,
    'a': np.array([0.2]),  # wave amplitudes
    'v': np.array([0.3]),  # wave frequencies
    'phis': np.array([0]),  # wave phases
}
env_params = update_derived_params_env(env_params)
g, rho_w, gamma_z, gamma_x, gamma_phi, a, v, phis, k_w, lambdas, n_waves = get_values_env(env_params)

# initialise WITT and buoy parameters
witt_params = {} # define dictionary of parameters
witt_size = 'small' # 'small' or 'large'
witt_params = update_witt_size(witt_size, witt_params)  # update WITT size
witt_size, m, m_s, m_pto, m_b, l, D, R, S, zeta0, c0, zeta, c, dtheta_PTO, thickness, M, J, total_weight, Ca, Cd = get_values_witt(witt_params)

# check key parameters
buoy_r_gyration, f_nat_p, f_nat_b = calc_nat_freqs(env_params, witt_params)
print_key_params(env_params, witt_params, f_nat_p, f_nat_b, buoy_r_gyration)
max_Fb, max_mooring_Fz = check_can_float(env_params, witt_params)

#%% run simulations

# initial conditions [x, dxdt, z, dzdt, theta, dthetadt, phi, dphidt]
r0 = [0, 0, 0, 0, 0, 0, 0, 0]

# setup time parameters
t_start = 0
t_current = t_start
t_end = 200
t_step = 0.1
t_eval = np.arange(t_start, t_end, t_step)
t_span = (t_start, t_end)

## choose independent variable
amplitudes = [0.1, 0.3, 0.6, 0.9, 1.2, 1.5] # wave amplitudes
frequencies = [0.15, 0.3, 0.44] #[[0.15, 0.3, 0.45]] #, 0.6, 0.9, 1.2, 1.5, 1.8] # wave frequencies (Hz)
high_frequencies = [0.44, 0.6, 0.7] #, 1.5] 
ballast_masses_small_D = [0, 10, 20, 30]
ballast_masses_large_D = [0, 10, 20, 30, 50, 100, 200, 300, 400]
buoy_diams_small = [0.45, 0.46, 0.475, 0.5, 0.75, 1] #, 1.25, 1.5] #[1, 1.25, 1.5] #[0.395, 
buoy_diams_small_unmoored = [0.42, 0.45, 0.5, 0.75, 1] #, 1.25, 1.5] #[1, 1.25, 1.5] #[0.395, 
buoy_diams_large = [1.25, 1.5, 1.75, 2, 2.25]
shell_masses_small_D = [5, 10, 15, 20, 30]
shell_masses_large_D = [10, 20, 30, 50, 100, 200, 300, 400]
pendulum_dampings = [0.1, 0.25, 0.5, 0.75, 1, 1.25]
witt_sizes = ['small','large']
added_mass_vals = [0, 0.25, 0.5, 0.75, 1]
Cd_vals = [0, 0.1, 0.25, 0.5, 0.75, 1]
n_moorings = [0, 1, 2]
depths = [5, 10, 20, 30, 40, 50, 75, 100]
mooring_alphas = np.array([0, 1/8, 1/4, 3/8, 1/2])*np.pi
mooring_X0s = np.array([0, 1/8, 1/4, 3/8])*np.pi
mooring_T0zs = [50, 100, 150, 200] #, 250, 300]
mooring_T0zs_large_D = [50, 100, 200, 300, 400]
mooring_T0xs = [50, 100, 150, 200, 250, 300]
lambda_m0s = [10, 50, 100, 150, 200] # mooring stiffness when taut (N/m)
stiffnesses_large_D = [10, 50, 100, 200, 300]#, 400]
lambda_fracs = [2, 5, 10, 20]
slack_allowed = [1, 2, 3, 4, 5, 7.5, 10]

# run simulation, args = (WITT_function, mooring_type, n_mooring, t_span, r0, t_eval, method)
params = [0.15, 0.3, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45] #frequencies
plot_labels = params
sols = []
Z0s = []
cairns_terms = []
PTO_terms = []
E_PTO = []
event_flag = False
pto_calc = True
save_data = True
optimised_params = True
# env_params = update_params_env(env_params,v=0.3)
# env_params = update_params_env(env_params,a=0.2)
if optimised_params:
    witt_params = update_params_witt(witt_params, D=1)
    witt_params = update_params_witt(witt_params, m_s=10)
    witt_params = update_params_witt(witt_params, m_b=0)
    mooring_params = update_params_mooring(mooring_params, lambda_m0=150)
    mooring_params = update_params_mooring(mooring_params, T0z=300)
    mooring_params = update_params_mooring(mooring_params, X0=(np.pi*3/8))
    mooring_params = update_params_mooring(mooring_params, alpha_m=0)
n_mooring = 2
# for _ in range(1):
for param in params:
    print(f'Running simulation for v = {param}...')
    ## update parameters
    # witt_params = update_params_witt(witt_params, D=param)
    env_params = update_params_env(env_params, v=param)
    # mooring_params = update_params_mooring(mooring_params, alpha_m=param)
    
    # check dtheta_PTO is correct
    if witt_params['dtheta_PTO'] != 4.5:
        print(f'Error: dtheta_PTO is not set correctly, it should be 4.5rad/s, but is actually {witt_params['dtheta_PTO']}rad/s')
        sys.exit(1)
    
    # run simulation
    sol, Z0 = run_sim(WITT_function=nonlinear_2D_WITT, mooring_type='simple_taut', n_mooring=n_mooring, t_span=t_span, r0=r0, t_eval=t_eval, method='LSODA', rtol=1e-3, atol=1e-6, env_params=env_params, witt_params=witt_params, mooring_params=mooring_params)
    sols.append(sol)
    Z0s.append(Z0)
    
    # recompute wave forcing terms
    A_vals = []
    sigma_vals = []
    for ti, xi in zip(sol.t, sol.y.T):  # sol.y.T gives state vectors
        x = xi[0] # index 0 = x position
        A, sigma = wave_forcing_cairns(ti, x, witt_params['R'], env_params['a'], env_params['v'], env_params['phis'], env_params['lambdas'], env_params['n_waves'])
        A_vals.append(A)
        sigma_vals.append(sigma)
    cairns_term_temp = np.vstack([A_vals, sigma_vals])  # shape (2, N)
    cairns_terms.append(cairns_term_temp)  # append to list of cairns terms
    
    # recompute power take-off force
    if pto_calc == True:
        Ftheta_vals = []
        Ftheta0_vals = []
        Power_vals = []
        for xi in sol.y.T:
            dthetadt = xi[5]  # index 5 = dtheta/dt
            angle = dthetadt
            Ftheta0 = -witt_params['c0'] * dthetadt  # natural damping force
            if np.abs(angle) > witt_params['dtheta_PTO']:
                Ftheta = -witt_params['c'] * angle + Ftheta0  # PTO damping force
                Power = np.abs((Ftheta - Ftheta0) * angle)  # Power = Force * angular velocity
            else:
                Ftheta = 0
                Power = 0
            Ftheta0_vals.append(Ftheta0)  # natural damping force
            Ftheta_vals.append(Ftheta)
            Power_vals.append(Power)
        PTO_terms_temp = np.vstack([Ftheta0_vals, Ftheta_vals, Power_vals]) # shape (2, N)
        PTO_terms.append(PTO_terms_temp) # append to list of power terms, one array in list per simulation
        
        # calculate total energy
        E_PTO.append(trapezoid(Power_vals, sol.t))  # or cumtrapz for time-resolved energy

print('Simulations completed.')
print(f'Energy extracted by PTO, Joules: {E_PTO}, for params {params}')

# append theta+phi to solutions
for i,sol in enumerate(sols):
    new_var = sol.y[4] + sol.y[6]
    sol.y = np.vstack((sol.y, new_var))
    sols[i] = sol

#%% perform frequency analysis
def frequency_analysis(sol, id):
    # use second half of signal to avoid transient behavior
    N_total = len(sol.t)
    half_idx = N_total // 2
    t_segment = sol.t[half_idx:]
    signal_segment = sol.y[id, half_idx:] - np.mean(sol.y[id, half_idx:])  # remove DC offset
    
    # compute FFT
    fft_result = fft(signal_segment)
    freqs = fftfreq(len(t_segment), t_step)
    
    # return positive frequencies and amplitudes
    positive_freqs = freqs[:len(freqs)//2]
    amplitudes = (2.0 / len(t_segment)) * np.abs(fft_result[:len(freqs)//2])
    return positive_freqs, amplitudes

# perform frequency analysis on requested IDs
sols_freqs = []
sols_amplitudes = []
ids = (0,1,2,3,4,5,6,7,8) # id = index to select variable from state vector ([x, dx, z, dz, theta, dtheta, phi, dphi, theta+phi] --> 0, 1, 2, 3, 4, 5, 6, 7, 8)
for sol in sols:
    # initialise temp 2D arrays to store results for each sol
    freqs_temp = []
    amplitudes_temp = []
    
    for id in ids:
        freqs, amps = frequency_analysis(sol, id)
        freqs_temp.append(freqs)
        amplitudes_temp.append(amps)
    
    # convert lists into 2D arrays (shape: 8 x Nfreqs)
    freqs_temp = np.array(freqs_temp)
    amplitudes_temp = np.array(amplitudes_temp)
    
    # append arrays into master lists
    sols_freqs.append(freqs_temp)
    sols_amplitudes.append(amplitudes_temp)

#%% initial plots to check results

# plot dtheta as a test for PTO threshold
plt.figure(figsize=(8, 5))
var_id = 5
for i,sol in enumerate(sols):
    label = plot_labels[i]
    plt.plot(sol.t, sol.y[var_id], label=label) #, label='n_mooring = 2')
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
    plt.plot(sol.t, sol.y[var_id], label=label) #, label='n_mooring = 2')
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
    plt.plot(sol.t, sol.y[var_id], label=label)
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
    plt.plot(sol.t, sol.y[var_id]-Z0s[i], label=label)
plt.title('Vertical position')
plt.ylabel('Depth (z-Z0) (m)')
plt.xlabel('Time (s)')
plt.legend()
plt.grid()
plt.xlim([t_start, t_end])
plt.subplot(2, 1, 2)
for i,sol in enumerate(sols):
    label = plot_labels[i]
    plt.plot(sol.t, cairns_terms[i][0,:], label=label)
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
    plt.plot(sol.t, sol.y[var_id], label=label)
plt.title('Pendulum angle (local axes)')
plt.ylabel('theta (rad)')
plt.xlabel('Time (s)')
plt.grid()
plt.legend()
plt.xlim([t_start, t_end])
plt.subplot(3, 1, 2)
var_id = 8
for i,sol in enumerate(sols):
    label = plot_labels[i]
    # time_temp, angle_temp = wrap_angles(sol.t, sol.y[var_id])  # wrap angles to [-pi, pi]
    # plt.plot(time_temp, angle_temp, label=label)
    plt.plot(sol.t, sol.y[var_id], label=label)
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
    plt.plot(sol.t, sol.y[var_id], label=label)
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
    plt.plot(sol.t, sol.y[var_id], label=label)
plt.title('Buoy angle')
plt.ylabel('phi (rad)')
plt.xlabel('Time (s)')
plt.legend()
plt.grid()
plt.xlim([t_start, t_end])
plt.subplot(2, 1, 2)
for i,sol in enumerate(sols):
    label = plot_labels[i]
    plt.plot(sol.t, cairns_terms[i][1,:], label=label)
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
    plt.plot(freqs[var_id, :], sols_amplitudes[i][var_id, :], label=label)
plt.title('Horizontal position')
plt.ylabel('Amplitudes')
plt.grid()
plt.xlim([0, 2])
plt.legend()
plt.subplot(2, 1, 2)
var_id = 2
for i,freqs in enumerate(sols_freqs):
    label = plot_labels[i]
    plt.plot(freqs[var_id, :], sols_amplitudes[i][var_id, :], label=label)
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
    plt.plot(freqs[var_id, :], sols_amplitudes[i][var_id, :], label=label)
plt.title('Pendulum angle, theta (local)')
plt.ylabel('Amplitudes')
plt.grid()
plt.xlim([0, 2])
plt.legend()
plt.subplot(3, 1, 2)
var_id = 8
for i,freqs in enumerate(sols_freqs):
    label = plot_labels[i]
    plt.plot(freqs[var_id, :], sols_amplitudes[i][var_id, :], label=label)
plt.title('Pendulum angle, theta+phi (global)')
plt.ylabel('Amplitudes')
plt.xlim([0, 2])
plt.grid()
plt.subplot(3, 1, 3)
var_id = 6
for i,freqs in enumerate(sols_freqs):
    label = plot_labels[i]
    plt.plot(freqs[var_id, :], sols_amplitudes[i][var_id, :], label=label)
plt.title('Buoy angle, phi')
plt.ylabel('Amplitudes')
plt.xlabel('Frequencies (Hz)')
plt.xlim([0, 2])
plt.grid()
plt.tight_layout()
# plt.savefig('02 Modelling/03 Summer Delivery/Single WITT - theta phi freqs.pdf')
plt.show()

if pto_calc == True:
    # plot natural damping (friction), applied torque, power using PTO_terms = [friction, torque, power]
    plt.figure(figsize=(8, 5))
    plt.subplot(3, 1, 1) # plot friction torque
    var_id = 0
    for i,sol in enumerate(sols):
        label = plot_labels[i]
        plt.plot(sol.t, PTO_terms[i][var_id,:], label=label)
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
        plt.plot(sol.t, PTO_terms[i][var_id,:], label=label)
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
        plt.plot(sol.t, PTO_terms[i][var_id,:], label=label)
    plt.title('PTO power')
    plt.ylabel('Power (W)')
    plt.xlabel('Time (s)')
    plt.grid()
    plt.legend()
    plt.xlim([t_start, t_end])
    plt.tight_layout()
    # plt.savefig('02 Modelling/03 Summer Delivery/Single WITT - theta phi.pdf')
    plt.show()


#%% save data/results for plotting elsewhere
import os

if save_data:
    output_folder = "02 Modelling/03 Summer Delivery/Results/Single WITT/Optimised Moored Power Freqs"  # set folder path
    os.makedirs(output_folder, exist_ok=True) # create the folder if it doesn't exist
    print(f'Saving data to {output_folder}')
    
    # want to save: sols, Z0, cairns_terms, PTO_terms
    sols_3d = np.array(sols) # convert sols to a 3D array
    np.savez(os.path.join(output_folder, "sim_outputs.npz"), sols_3d=sols_3d)
    np.savez(os.path.join(output_folder, "Z0s.npz"), Z0s=np.array(Z0s))
    np.savez(os.path.join(output_folder, "cairns_terms.npz"), cairns_terms=np.array(cairns_terms))
    np.savez(os.path.join(output_folder, "PTO_terms.npz"), PTO_terms=np.array(PTO_terms))
    np.savez(os.path.join(output_folder, "WITT_params.npz"), WITT_params=np.array([witt_params['D'],witt_params['l'],n_mooring]))
    np.savez(os.path.join(output_folder, "loop_params.npz"), loop_params=np.array(params))
    np.savez(os.path.join(output_folder, "wave_params.npz"), n_waves=env_params['n_waves'],a=env_params['a'],v=env_params['v'],phis=env_params['phis'],lambdas=env_params['lambdas'],h=mooring_params['h'])
    np.savez(os.path.join(output_folder, "freq_results.npz"), freqs=sols_freqs,amps=sols_amplitudes)
    np.savez(os.path.join(output_folder, "mooring_params.npz"), mooring_params=np.array([mooring_params['X0'],mooring_params['alpha_m'],n_mooring]))
else:
    print('Simulation data not saved')
