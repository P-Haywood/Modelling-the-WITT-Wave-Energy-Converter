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
# v1.0 - (retired on 07/08)
# Created on 23/07 using code from '2D WITT ODE v2.1.py', this code models a single WITT buoy system but in a neater and cleaner way...
# ...with things turned into functions, corrections made, old code elements omitted etc
#
# Key things to NOTE: - T0 is per mooring line, so vertical tension is n_mooring*T0*cos(X0)
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
#                 - added event detection functionality for PTO force, 07/08
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
    h, alpha_m, X0, T0, lambda_m0, lambda_m, slack_allowed, Kx, Kz, Kphi, Kxphi = get_values_mooring(mooring_params) # args: h, alpha_m, X0, T0, lambda_m0, lambda_m, slack_allowed
    t_start, t_end = t_span
    
    buoy_r_gyration, f_nat_p, f_nat_b = calc_nat_freqs(env_params, witt_params)
    print_key_params(env_params, witt_params, f_nat_p, f_nat_b, buoy_r_gyration)
    max_Fb, max_mooring_Fz = check_can_float(env_params, witt_params)
    
    # initialise mooring
    if n_mooring == 0:
        print('No mooring lines...')
        mooring_params_calced = []
        
        # equilibrium depth
        def f(Z0): # -TODO: make this more robust? can get issues if initial guess isn't close to the root I want (i.e. want a solution < R, otherwise S_offset < 0 which is BAD!!)
            return (R + Z0)**2 * (2*R - Z0) - 3*M/(rho_w*np.pi)
        initial_guess = 0.2*R
        Z0 = np.ndarray.item(fsolve(f, initial_guess))
        print(f'Z0 solution found as: {Z0} m (positive downwards)')
        eq_Fb = 1/3 * rho_w * g * np.pi * (R + Z0)**2 * (2*R - Z0) * np.cos(0) # buoyancy force at equilibrium
        print(f'Buoyancy force at equilibrium (Z0) is: {eq_Fb} N, weight is: {total_weight} N')
        S_offset = np.pi * (R**2 - Z0**2) # area at waterline in equilibrium
        print('S_offset is: ', S_offset, 'm^2')
        
    else:
        print('Initialising mooring...')
        Fzm_0 = n_mooring*T0*np.cos(X0) # initial vertical mooring force
        
        if mooring_type == 'simplest_mooring':
            print('Using the simplest mooring model...')
            
            # calculate equilibrium depth
            Fzm_0 = T0
            def f(Z0): # -TODO: make this more robust? can get issues if initial guess isn't close to the root I want (i.e. want a solution < R, otherwise S_offset < 0 which is BAD!!)
                return (R + Z0)**2 * (2*R - Z0) - 3/(rho_w*np.pi*g) * (total_weight + Fzm_0)
            initial_guess = 0 #0.2*R
            Z0 = np.ndarray.item(fsolve(f, initial_guess))
            print(f'Z0 solution found as: {Z0} m (positive downwards)')
            
            # check buoyancy vs weight + initial vertical mooring tension
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
            
            # print mooring parameters
            print (f'Mooring parameters: T0 {T0}, Kxx {Kx}, Kzz {Kz}, Kphi {Kphi}, Kxphi {Kxphi}')
            mooring_params_calced = [] # initialise empty list so code runs
            
            # calculate initial mooring forces
            Fx, Fz, Mphi = simplest_mooring(0, 0, 0, T0, Kx, Kz, Kphi, Kxphi, n_mooring)
            print(f'Initial mooring forces: Fx {Fx}, Fz {Fz}, Mphi {Mphi}')
            
            # check vertical mooring force matches that from T0
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
        elif mooring_type == 'simple_taut':
            print('Using simple taut elastic mooring model...')
            
            # calculate equilibrium depth
            def f(Z0): # -TODO: make this more robust? can get issues if initial guess isn't close to the root I want (i.e. want a solution < R, otherwise S_offset < 0 which is BAD!!)
                return (R + Z0)**2 * (2*R - Z0) - 3/(rho_w*np.pi*g) * (total_weight + Fzm_0)
            initial_guess = 0 #0.2*R
            Z0 = np.ndarray.item(fsolve(f, initial_guess))
            print(f'Z0 solution found as: {Z0} m (positive downwards)')
            
            # check buoyancy vs weight + initial vertical mooring tension
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
            
            # calculate other parameters
            h1 = h - R*np.cos(alpha_m) - Z0 # initial depth of mooring point on hull, = water depth - mooring position from centre - Z0
            h2 = h1*np.tan(X0) # initial projected length of mooring line on horizontal plane (m)
            L0_prestressed = np.sqrt(h1**2 + h2**2) # initial pre-stressed length of mooring line
            L0_true = L0_prestressed - T0/lambda_m0 # calculate true natural length of mooring line based on tension and prestressed length
            mooring_params_calced = [h1, h2, L0_true]

            # print mooring parameters
            print (f'Mooring parameters: T0 {T0}, X0 (initial angle) {X0}, lambda_m0 {lambda_m0}, h1 {h1}, h2 {h2}, L0_prestressed {L0_prestressed}, L0_true {L0_true}')
            
            # calculate initial mooring forces
            Fx, Fz, Mphi = simple_taut_mooring(0, 0, 0, R, n_mooring, Z0, alpha_m, h, lambda_m0, mooring_params_calced)  # initial conditions for mooring forces
            print(f'Initial mooring forces: Fx {Fx}, Fz {Fz}, Mphi {Mphi}')
            
            # check vertical mooring force matches that from T0
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

        elif mooring_type == 'piecewise': #-TODO: update this to match simple_taut, i.e. then find initial mooring forces etc 
            print('Using piecewise mooring model...')
            
            # calculate equilibrium depth
            def f(Z0): # -TODO: make this more robust? can get issues if initial guess isn't close to the root I want (i.e. want a solution < R, otherwise S_offset < 0 which is BAD!!)
                return (R + Z0)**2 * (2*R - Z0) - 3/(rho_w*np.pi*g) * (total_weight + Fzm_0)
            initial_guess = 0 #0.2*R
            Z0 = np.ndarray.item(fsolve(f, initial_guess))
            print(f'Mooring Z0 solution found as: {Z0} m (positive downwards)') 
            
            # check buoyancy vs weight + initial vertical mooring force 
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

# function for nonlinear 2D (planar) WITT device
prev_dxdt2 = 0
def nonlinear_2D_WITT(t, r, mooring_type, n_mooring, Z0, mooring_params_calced, env_params, witt_params, mooring_params, pto_flag):
    """nonlinear 2D ODE function for a WITT device in a spherical buoy

    Args:
        t (float): time
        r (list): state vector [x, dx/dt, z, dz/dt, theta, dtheta/dt]

    Returns:
        list: derivatives [dx/dt, d2x/dt2, dz/dt, d2z/dt2, dtheta/dt, d2theta/dt2]
    """
    x, dxdt, z, dzdt, theta, dthetadt, phi, dphidt = r
    global prev_dxdt2
    
    # read in parameters# read in parameters from dictionaries
    g, rho_w, gamma_z, gamma_x, gamma_phi, a, v, phis, k_w, lambdas, n_waves = get_values_env(env_params)
    witt_size, m, m_s, m_pto, m_b, l, D, R, S, zeta0, c0, zeta, c, dtheta_PTO, thickness, M, J, total_weight, Ca, Cd = get_values_witt(witt_params)
    h, alpha_m, X0, T0, lambda_m0, lambda_m, slack_allowed, Kx, Kz, Kphi, Kxphi = get_values_mooring(mooring_params) # args: h, alpha_m, X0, T0, lambda_m0, lambda_m, slack_allowed
    
    # dynamic mass matrix (dependent on theta)
    Cm = 1 + Ca # mass coefficient, i.e. normal mass + added mass
    Mmat = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, Cm*M + m, 0, 0, 0, m*l*np.cos(theta+phi), 0, m*l*np.cos(theta+phi)],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, Cm*M + m, 0, m*l*np.sin(theta+phi), 0, m*l*np.sin(theta+phi)],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, m*l*np.cos(theta+phi), 0, m*l*np.sin(theta+phi), 0, m*l**2, 0, m*l**2],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, m*l*np.cos(theta+phi), 0, m*l*np.sin(theta+phi), 0, m*l**2, 0, J+m*l**2]
    ])
    
    # wave forcing
    A, sigma_ave = wave_forcing_cairns(t, x, R, a, v, phis, lambdas, n_waves)  # wave forcing from Cairns et al approach
    
    # Buoyancy force - constrain h_s between 0 and D then calculate buoyancy force, centre of buoyancy, and moment - TODO: make this more robust? smaller wavelengths?
    h_s = R + (Z0 + A - z)*np.cos(sigma_ave) #*np.cos(theta)
    # F_b, Moment_b = buoyancy_force(h_s, rho_w, g, R, sigma_ave) # -NOTE: Don't use this, as slows down code MASSIVELY
    if h_s <= 0:
        h_s = 0
        F_b = 0
        Moment_b = 0
    elif h_s >= D:
        h_s = D
        F_b = 4/3*rho_w * g * np.pi * R**3 # buoyancy force at full submersion
        CoB = 0 #3/4 * (2*R - h_s)**2 / (3*R - h_s)
        Moment_b = 0 #F_b * np.sin(sigma_ave) * CoB
    else:
        CoB = 3/4 * (2*R - h_s)**2 / (3*R - h_s)
        F_b = 1/3 * rho_w * g * np.pi * h_s**2 * (3*R - h_s) * np.cos(sigma_ave)
        Moment_b = F_b * np.sin(sigma_ave) * CoB
    
    # Horizontal hydrostatic force - TODO: make this more robust, i.e. if intersection is beyond centre then need to use area of piR**2 and centroid at centre
    F_hydro, M_hydro = 0, 0
    F_hydrodyn, M_hydrodyn = 0, 0
    # F_hydro = horizontal_hydrostatic_force(h_s, x, z, R, Z0, t, a, v, lambdas, phis, n_waves, sigma_ave, A) # -NOTE: Don't use this, as throws errors
    if h_s <= 0:
        F_hydro = 0
    elif h_s >= D:
        F_hydro = 0
    else:
        # find coordinates at middle of buoy cross section made by water surface at amplitude A and angle sigma
        x_m = x - np.abs((h_s-R)*np.sin(sigma_ave))
        z_m = z + (h_s-R)*np.cos(sigma_ave)
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
        M_hydro_left = F_hydro_left*(z_l-z_cl)
        M_hydro_right = F_hydro_right*(z_r-z_cr)
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
        
        # # Hydrodynamic forces - Morison's equation - NOTE: legacy code, no longer in use as doesn't work/run well, 05/08
        # area_waterplane = np.pi*R_s**2
        # Ca = 0.5
        # z_upper = z_l-Z0 # z limits are relative to quiescent water surface
        # z_lower = z-R-Z0
        # F_hydrodyn_l, M_hydrodyn_l = morison_force(t, x, dxdt, prev_dxdt2, area_waterplane, z_upper, z_lower, rho_w, Ca, Cd, h, k_w, v, R, z, Z0)
        # z_upper = z_r-Z0
        # F_hydrodyn_r, M_hydrodyn_r = morison_force(t, x, dxdt, prev_dxdt2, area_waterplane, z_upper, z_lower, rho_w, Ca, Cd, h, k_w, v, R, z, Z0)
        # F_hydrodyn = F_hydrodyn_l - F_hydrodyn_r
        # M_hydrodyn = M_hydrodyn_l - M_hydrodyn_r
        # # print(F_hydrodyn, M_hydrodyn)
        # # # set to 0 for testing
        # # F_hydrodyn, M_hydrodyn = 0, 0
    
    # Drag force
    arg = np.clip((Z0-z+A)/R, -1, 1)
    delta_drag = 2*np.pi - 2*np.arccos(arg) # find angle of segment above water, then subtract from 2pi for angle of segment below
    ref_area_x = R**2/2 * (delta_drag - np.sin(delta_drag)) # average vertical submerged area
    ref_area_x = max(ref_area_x, 0.0)
    ref_area_z = np.cos(sigma_ave)*np.pi*(R**2 - (h_s-R)**2) # projected area (in xy plane) between 2 intersection points
    ref_area_z = max(ref_area_z, 0.0)
    if np.abs(delta_drag) < 1e-7:
        z_drag = R
    elif np.abs(2*np.pi - delta_drag) < 1e-7:
        z_drag = 0
    else:
        z_drag = (4*R*np.sin(delta_drag/2)**3) / (3*(delta_drag-np.sin(delta_drag))) # centroid of drag reference area (for force application/moment)
    # ref_area = np.pi*(R**2 - (h_s-R)**2) # cross sectional area of buoy at waterplane
    dxdt_rel = dxdt # -TODO: update these to actually be relative velocity of water compared to buoy??
    dzdt_rel = dzdt
    dphidt_rel = dphidt
    # print(f'Drag terms: arg {arg}, {delta_drag}, ref_area {ref_area}, z_drag {z_drag}')
    F_drag_x = -0.5 * rho_w * Cd * ref_area_x * (dxdt_rel) * np.abs(dxdt_rel)
    F_drag_z = -0.5 * rho_w * Cd * ref_area_z * (dzdt_rel) * np.abs(dzdt_rel)
    M_drag_x = F_drag_x*z_drag
    M_drag_phi = -0.5 * rho_w * Cd * ref_area_x * (dphidt_rel) * np.abs(dphidt_rel)
    # print(f'Drag forces: Fx {F_drag_x}, Mx {M_drag_x}')
    drag_error_array = np.array([arg, delta_drag, ref_area_x, z_drag, F_drag_x, M_drag_x])
    if not np.all(np.isfinite(drag_error_array)):
        print("Error: array contains NaN or Inf")
        print("Values:", drag_error_array)
    
    # mooring forces
    if n_mooring == 0:
        Fxm, Fzm, Fphim = 0, 0, 0
    else:
        if mooring_type == 'simplest_mooring':
            Fxm, Fzm, Fphim = simplest_mooring(x, z, phi, T0, Kx, Kz, Kphi, Kxphi, n_mooring)
        elif mooring_type == 'simple_taut':
            Fxm, Fzm, Fphim = simple_taut_mooring(x, z, phi, R, n_mooring, Z0, alpha_m, h, lambda_m0, mooring_params_calced)
        elif mooring_type == 'piecewise':
            Fxm, Fzm, Fphim = piecewise_mooring(x, z, phi, R, n_mooring, Z0, alpha_m, h, lambda_m0, lambda_m, mooring_params_calced)
    # elif mooring_type == 'catenary': # add these later for K11 stiffnesses etc if wanted
    # elif mooring_type == 'taut':
    
    # non-conservative forces
    Fx = Fxm + F_hydro + F_hydrodyn + F_drag_x #+ Ax*np.cos(w*t)
    Fg = (M+m)*g #+ m*l*dthetadt**2*np.cos(theta)
    Fz = Fzm + F_drag_z # + Ax*0.5*np.cos(w*t)
    Ftheta = -c0*dthetadt
    Fphi = Fphim + M_hydro + M_hydrodyn + M_drag_x + M_drag_phi #+ Ax*0.1*np.cos(w*t)
    
    # power take-off
    if not event_flag and pto_calc == True:
        if np.abs(dthetadt) > dtheta_PTO:
            epsilon = 1  # smoothing factor, smaller is sharper, aim for 0.05*dtheta_PTO or smaller
            activation = 0.5 * (1 + np.tanh((np.abs(dthetadt) - dtheta_PTO) / epsilon))
            c_eff = c0 + (c - c0) * activation
            Ftheta = -c_eff * dthetadt
            # Ftheta = -c*dthetadt  # PTO force in theta direction
        else:
            Ftheta = -c0*dthetadt
    else:
        Ftheta = -c0*dthetadt
    
    # equations of motion
    dxdt2 = Fx + m*l*(dthetadt+dphidt)**2*np.sin(theta+phi) #- gamma_x*dxdt # - K11*x - K12*phi
    dzdt2 = Fz - m*l*(dthetadt+dphidt)**2*np.cos(theta+phi) - g*(M+m) + F_b - gamma_z*dzdt # - K33*z
    dthetadt2 = Ftheta - m*g*l*np.sin(theta+phi)
    dphidt2 = Fphi + Moment_b - m*g*l*np.sin(theta+phi) - R*m_b*g*np.sin(phi) - gamma_phi*dphidt # - K22*phi - K12*x
    
    # f(r,t) vector
    fvec = np.array([
        dxdt,
        dxdt2,
        dzdt,        # force dz/dt = 0
        dzdt2,        # force dz/dt2 = 0
        dthetadt,
        dthetadt2,
        dphidt,
        dphidt2
    ])
    
    # solve for dr/dt
    rdot = np.linalg.solve(Mmat, fvec)
    prev_dxdt2 = rdot[1]
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
    global prev_dxdt2
    
    # read in parameters# read in parameters from dictionaries
    g, rho_w, gamma_z, gamma_x, gamma_phi, a, v, phis, k_w, lambdas, n_waves = get_values_env(env_params)
    witt_size, m, m_s, m_pto, m_b, l, D, R, S, zeta0, c0, zeta, c, dtheta_PTO, thickness, M, J, total_weight, Ca, Cd = get_values_witt(witt_params)
    h, alpha_m, X0, T0, lambda_m0, lambda_m, slack_allowed, Kx, Kz, Kphi, Kxphi = get_values_mooring(mooring_params) # args: h, alpha_m, X0, T0, lambda_m0, lambda_m, slack_allowed
    
    # dynamic mass matrix (dependent on theta)
    Cm = 1 + Ca # mass coefficient, i.e. normal mass + added mass
    Mmat = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, Cm*M + m, 0, 0, 0, m*l*np.cos(theta+phi), 0, m*l*np.cos(theta+phi)],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, Cm*M + m, 0, m*l*np.sin(theta+phi), 0, m*l*np.sin(theta+phi)],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, m*l*np.cos(theta+phi), 0, m*l*np.sin(theta+phi), 0, m*l**2, 0, m*l**2],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, m*l*np.cos(theta+phi), 0, m*l*np.sin(theta+phi), 0, m*l**2, 0, J+m*l**2]
    ])
    
    # wave forcing
    A, sigma_ave = wave_forcing_cairns(t, x, R, a, v, phis, lambdas, n_waves)  # wave forcing from Cairns et al approach
    
    # Buoyancy force - constrain h_s between 0 and D then calculate buoyancy force, centre of buoyancy, and moment - TODO: make this more robust? smaller wavelengths?
    h_s = R + (Z0 + A - z)*np.cos(sigma_ave) #*np.cos(theta)
    # F_b, Moment_b = buoyancy_force(h_s, rho_w, g, R, sigma_ave) # -NOTE: Don't use this, as slows down code MASSIVELY
    if h_s <= 0:
        h_s = 0
        F_b = 0
        Moment_b = 0
    elif h_s >= D:
        h_s = D
        F_b = 4/3*rho_w * g * np.pi * R**3 # buoyancy force at full submersion
        CoB = 0 #3/4 * (2*R - h_s)**2 / (3*R - h_s)
        Moment_b = 0 #F_b * np.sin(sigma_ave) * CoB
    else:
        CoB = 3/4 * (2*R - h_s)**2 / (3*R - h_s)
        F_b = 1/3 * rho_w * g * np.pi * h_s**2 * (3*R - h_s) * np.cos(sigma_ave)
        Moment_b = F_b * np.sin(sigma_ave) * CoB
    
    # Horizontal hydrostatic force - TODO: make this more robust, i.e. if intersection is beyond centre then need to use area of piR**2 and centroid at centre
    F_hydro, M_hydro = 0, 0
    F_hydrodyn, M_hydrodyn = 0, 0
    # F_hydro = horizontal_hydrostatic_force(h_s, x, z, R, Z0, t, a, v, lambdas, phis, n_waves, sigma_ave, A) # -NOTE: Don't use this, as throws errors
    if h_s <= 0:
        F_hydro = 0
    elif h_s >= D:
        F_hydro = 0
    else:
        # find coordinates at middle of buoy cross section made by water surface at amplitude A and angle sigma
        x_m = x - np.abs((h_s-R)*np.sin(sigma_ave))
        z_m = z + (h_s-R)*np.cos(sigma_ave)
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
        M_hydro_left = F_hydro_left*(z_l-z_cl)
        M_hydro_right = F_hydro_right*(z_r-z_cr)
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
        
        # # Hydrodynamic forces - Morison's equation - NOTE: legacy code, no longer in use as doesn't work/run well, 05/08
        # area_waterplane = np.pi*R_s**2
        # Ca = 0.5
        # z_upper = z_l-Z0 # z limits are relative to quiescent water surface
        # z_lower = z-R-Z0
        # F_hydrodyn_l, M_hydrodyn_l = morison_force(t, x, dxdt, prev_dxdt2, area_waterplane, z_upper, z_lower, rho_w, Ca, Cd, h, k_w, v, R, z, Z0)
        # z_upper = z_r-Z0
        # F_hydrodyn_r, M_hydrodyn_r = morison_force(t, x, dxdt, prev_dxdt2, area_waterplane, z_upper, z_lower, rho_w, Ca, Cd, h, k_w, v, R, z, Z0)
        # F_hydrodyn = F_hydrodyn_l - F_hydrodyn_r
        # M_hydrodyn = M_hydrodyn_l - M_hydrodyn_r
        # # print(F_hydrodyn, M_hydrodyn)
        # # # set to 0 for testing
        # # F_hydrodyn, M_hydrodyn = 0, 0
    
    # Drag force
    arg = np.clip((Z0-z+A)/R, -1, 1)
    delta_drag = 2*np.pi - 2*np.arccos(arg) # find angle of segment above water, then subtract from 2pi for angle of segment below
    ref_area = R**2/2 * (delta_drag - np.sin(delta_drag)) # average vertical submerged area
    ref_area = max(ref_area, 0.0)
    if np.abs(delta_drag) < 1e-7:
        z_drag = R
    elif np.abs(2*np.pi - delta_drag) < 1e-7:
        z_drag = 0
    else:
        z_drag = (4*R*np.sin(delta_drag/2)**3) / (3*(delta_drag-np.sin(delta_drag))) # centroid of drag reference area (for force application/moment)
    # ref_area = np.pi*(R**2 - (h_s-R)**2) # cross sectional area of buoy at waterplane
    dxdt_rel = dxdt # -TODO: update these to actually be relative velocity of water compared to buoy??
    dzdt_rel = dzdt
    # print(f'Drag terms: arg {arg}, {delta_drag}, ref_area {ref_area}, z_drag {z_drag}')
    F_drag_x = -0.5 * rho_w * Cd * ref_area * (dxdt_rel) * np.abs(dxdt_rel)
    F_drag_z = 0 #-0.5 * rho_w * Cd * ref_area * (dzdt_rel) * np.abs(dzdt_rel)
    M_drag_x = F_drag_x*z_drag
    # print(f'Drag forces: Fx {F_drag_x}, Mx {M_drag_x}')
    drag_error_array = np.array([arg, delta_drag, ref_area, z_drag, F_drag_x, M_drag_x])
    if not np.all(np.isfinite(drag_error_array)):
        print("Error: array contains NaN or Inf")
        print("Values:", drag_error_array)
    
    # mooring forces
    if n_mooring == 0:
        Fxm, Fzm, Fphim = 0, 0, 0
    else:
        if mooring_type == 'simplest_mooring':
            Fxm, Fzm, Fphim = simplest_mooring(x, z, phi, T0, Kx, Kz, Kphi, Kxphi, n_mooring)
        elif mooring_type == 'simple_taut':
            Fxm, Fzm, Fphim = simple_taut_mooring(x, z, phi, R, n_mooring, Z0, alpha_m, h, lambda_m0, mooring_params_calced)
        elif mooring_type == 'piecewise':
            Fxm, Fzm, Fphim = piecewise_mooring(x, z, phi, R, n_mooring, Z0, alpha_m, h, lambda_m0, lambda_m, mooring_params_calced)
    # elif mooring_type == 'catenary': # add these later for K11 stiffnesses etc if wanted
    # elif mooring_type == 'taut':
    
    # non-conservative forces
    Fx = Fxm + F_hydro + F_hydrodyn + F_drag_x #+ Ax*np.cos(w*t)
    Fg = (M+m)*g #+ m*l*dthetadt**2*np.cos(theta)
    Fz = Fzm + F_drag_z # + Ax*0.5*np.cos(w*t)
    Ftheta = -c0*dthetadt
    Fphi = Fphim + M_hydro + M_hydrodyn + M_drag_x #+ Ax*0.1*np.cos(w*t)
    
    # power take-off
    if np.abs(dthetadt) > dtheta_PTO:
        # epsilon = 0.1*dtheta_PTO**2 #0.04*dethat_PTO**3 #0.5  # smoothing factor, smaller is sharper, aim for 0.05*dtheta_PTO or smaller
        # # using dtheta_PTO**2 because when threshold is low the curve can be sharper, yet for higher thresholds the curve NEEDS to be smoother
        # activation = 0.5 * (1 + np.tanh((np.abs(dthetadt) - dtheta_PTO) / epsilon))
        # c_eff = c0 + (c - c0) * activation
        # Ftheta = -c_eff * dthetadt
        Ftheta = -c*dthetadt  # PTO force in theta direction
    
    # equations of motion
    dxdt2 = Fx + m*l*(dthetadt+dphidt)**2*np.sin(theta+phi) #- gamma_x*dxdt # - K11*x - K12*phi
    dzdt2 = Fz - m*l*(dthetadt+dphidt)**2*np.cos(theta+phi) - g*(M+m) + F_b - gamma_z*dzdt # - K33*z
    dthetadt2 = Ftheta - m*g*l*np.sin(theta+phi)
    dphidt2 = Fphi + Moment_b - m*g*l*np.sin(theta+phi) - R*m_b*g*np.sin(phi) - gamma_phi*dphidt # - K22*phi - K12*x
    
    # f(r,t) vector
    fvec = np.array([
        dxdt,
        dxdt2,
        dzdt,        # force dz/dt = 0
        dzdt2,        # force dz/dt2 = 0
        dthetadt,
        dthetadt2,
        dphidt,
        dphidt2
    ])
    
    # solve for dr/dt
    rdot = np.linalg.solve(Mmat, fvec)
    prev_dxdt2 = rdot[1]
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
    A_temp = np.zeros(n_waves)
    for i in range(n_waves):
        A_temp[i] = a[i] * np.sin(2*np.pi * (v[i]*t - x/lambdas[i]) + phis[i])
    A = np.sum(A_temp)
    
    # surface angle
    sigma_list = []
    for x_temp in np.array([x-R, x, x+R]):
        sigma_temp = np.zeros(n_waves)
        for i in range(n_waves):
            sigma_temp[i] = 2*np.pi*a[i]/lambdas[i] * np.cos(2*np.pi*(v[i]*t - x_temp/lambdas[i]) + phis[i])
        sigma_list.append(-np.arctan(np.sum(sigma_temp)))
    sigma_ave = np.mean(sigma_list)
    
    return A, sigma_ave

def buoyancy_force(h_s, rho_w, g, R, sigma_ave):
    # if h_s <= 0:
    #     h_s = 0
    #     F_b = 0
    #     Moment_b = 0
    # elif h_s >= D:
    #     h_s = D
    #     F_b = 4/3*rho_w * g * np.pi * R**3 # buoyancy force at full submersion
    #     CoB = 0 #3/4 * (2*R - h_s)**2 / (3*R - h_s)
    #     Moment_b = 0 #F_b * np.sin(sigma_ave) * CoB
    # else:
    h_s = np.clip(h_s, 0.0, D)

    if h_s == 0:
        F_b = 0
        Moment_b = 0
    elif h_s == D:
        F_b = 4/3 * rho_w * g * np.pi * R**3
        Moment_b = 0
    else:
        CoB = 3/4 * (2*R - h_s)**2 / (3*R - h_s)
        F_b = 1/3 * rho_w * g * np.pi * h_s**2 * (3*R - h_s) * np.cos(sigma_ave)
        Moment_b = F_b * np.sin(sigma_ave) * CoB
    
    return F_b, Moment_b

def morison_force(t, x, dxdt, dxdt2, Area, z_upper, z_lower, rho, Ca, Cd, h, k_w, v, R, z_b, Z0):
    omega = 2*np.pi*v
    
    def u(z):
        return (np.cosh(k_w * (h + z)) / np.cosh(k_w * h)) * np.cos(k_w * x - omega * t) #np.sin(2*np.pi * (v[i]*t - x/lambdas[i]) + phis[i])
    
    def dudt(z):
        return (np.cosh(k_w * (h + z)) / np.cosh(k_w * h)) * omega * np.sin(k_w * x - omega * t)
    
    def W(z): # width of a circle at a given z
        z_rel = z + Z0 - z_b # z relative to circle centroid, not relative to quiescent surface
        inside = max(R**2-z_rel**2, 0.0)
        return 2*np.sqrt(inside) # make sure W(z) is real and positive
    
    def u_area_integrand(z):
        return u(z) * W(z)
    
    def integrand(z):        
        dudt_z = dudt(z)
        W_z = W(z)
        
        fk = rho * W_z * dudt_z
        added_mass = rho * Ca * W_z * (dudt_z - dxdt2)
        
        return fk + added_mass
    
    def moment_density(z):
        return z * integrand(z)
    
    # find interial force terms
    eps = 1e-9 # use a slight numerical offset to prevent errors when inegrating to the bottom of the buoy
    f1, _ = quad(integrand, z_lower+eps, 0) 
    f2, _ = quad(integrand, 0, z_upper)
    force_inertial = f1 + f2
    m1, _ = quad(moment_density, z_lower+eps, 0)
    m2, _ = quad(moment_density, 0, z_upper)
    m_inertial = m1 + m2
    
    # find centre of pressure
    z_cp = m_inertial / force_inertial if force_inertial != 0 else np.nan
    
    # find lumped drag using average velocity 
    u_avg, _ = quad(u_area_integrand, z_lower, z_upper)
    u_avg /= Area
    drag = 0.5 * rho * Cd * Area * (u_avg - dxdt) * np.abs(u_avg - dxdt) # drag using cross sectional area of buoy along water plane
    
    # estimate mean velocity depth (at which drag acts)
    def rel_moment_density(z):
        return z * u_area_integrand(z)
    rel_moment, _ = quad(rel_moment_density, z_lower, z_upper)
    z_drag = rel_moment / ((z_upper-z_lower) * u_avg) if u_avg != 0 else np.nan
    
    # Sum forces and moments etc
    F_total = force_inertial + drag
    M_total = m_inertial + drag*z_drag
    
    return F_total, M_total

# define function for the simplest mooring
def simplest_mooring(x, z, phi, T0, Kx, Kz, Kphi, Kxphi, n_mooring):
    if n_mooring == 1:
        if x > 0:
            Fx = -Kx*x
            Mphi = -Kxphi*x
        else:
            Fx = 0
            Mphi = 0
        
        if phi > 0:
            Fx = Fx - Kxphi*phi
        else:
            Fx = Fx
        
        Fz = -T0 - Kz*z
        Mphi = Mphi - Kphi*phi
    elif n_mooring == 2:
        Fx = -Kx*x - Kxphi*phi
        Fz = T0 - Kz*z
        Mphi = -Kphi*phi - Kxphi*x
    
    return Fx, Fz, Mphi

# define function for simple_taut mooring forces
def simple_taut_mooring(x, z, phi, R, n_mooring, Z0, alpha_m, h, lambda_m0, mooring_params_calced):
    
    # read in params
    h1, h2, L0_true = mooring_params_calced
    
    # change in attachment point coordinates
    x_a10 = R*(1-np.sin(alpha_m))
    x_a1 = R*(1-np.cos(np.pi/2 - alpha_m + phi))
    dx_a1 = x + x_a1 - x_a10
    x_a20 = x_a10
    x_a2 = R*(1-np.cos(np.pi/2 - alpha_m - phi))
    dx_a2 = x - x_a2 + x_a20
    z_a10 = R*np.cos(alpha_m)
    z_a1 = R*np.sin(np.pi/2 - alpha_m + phi)
    dz_a1 = z - z_a1 + z_a10
    z_a20 = z_a10
    z_a2 = R*np.sin(alpha_m + phi)
    dz_a2 = z - z_a2 + z_a20
    
    # angles of mooring lines
    X1 = np.arctan((h2+dx_a1)/(h1+dz_a1)) #np.arctan((h2 + x_a1)/(h-Z0-z_a1))
    X2 = np.arctan((h2-dx_a2)/(h1+dz_a2)) #np.arctan((h2 - x_a2)/(h-Z0-z_a2))
    
    # calculate true extension
    delta_L1 = np.sqrt((h2 + dx_a1)**2 + (h1+dz_a1)**2) - L0_true  # effective change in length of mooring line for buoy 1
    delta_L2 = np.sqrt((h2 - dx_a2)**2 + (h1+dz_a2)**2) - L0_true  # effective change in length of mooring line for buoy 2
    
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

# define function for mooring forces
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
    x_a10 = R*(1-np.sin(alpha_m))
    x_a1 = R*(1-np.cos(np.pi/2 - alpha_m + phi))
    dx_a1 = x + x_a1 - x_a10
    x_a20 = x_a10
    x_a2 = R*(1-np.cos(np.pi/2 - alpha_m - phi))
    dx_a2 = x - x_a2 + x_a20
    z_a10 = R*np.cos(alpha_m)
    z_a1 = R*np.sin(np.pi/2 - alpha_m + phi)
    dz_a1 = z - z_a1 + z_a10
    z_a20 = z_a10
    z_a2 = R*np.sin(alpha_m + phi)
    dz_a2 = z - z_a2 + z_a20
    
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
    k_w = (2 * np.pi * v) ** 2 / g
    env_params['k_w'] = k_w
    env_params['lambdas'] = 2 * np.pi / k_w
    env_params['n_waves'] = len(env_params['a'])
    print(f"Wavelengths are: {env_params['lambdas']}")
    return env_params   

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
            J = R**2 * (m_b + (2/3) * m_s * ((1 - (R - thickness)**5 / R**5) / (1 - (R - thickness)**3 / R**3)))
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
    print(f'Buoy parameters: diameter {D}m, radius {R}m, shell mass {m_s}kg, ballast mass {m_b}kg, total mass (buoy+WITT) {M}kg, radius of gyration {buoy_r_gyration}m')
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

def get_values_mooring(mooring_params): # args: h, alpha_m, X0, T0, lambda_m0, lambda_m, slack_allowed, Kx, Kz, Kphi, Kxphi
    return (
        mooring_params['h'], mooring_params['alpha_m'], mooring_params['X0'],
        mooring_params['T0'], mooring_params['lambda_m0'], mooring_params['lambda_m'], mooring_params['slack_allowed'],
        mooring_params['Kx'], mooring_params['Kz'], mooring_params['Kphi'], mooring_params['Kxphi']
    )

def update_witt_size(witt_size, witt_params):
    if witt_size == 'small':
        witt_params.update({
                'witt_size': 'small',
                'm': 5.854,
                'm_s': 7.31,
                'm_pto': 2.61,
                'm_b': 10,
                'l': 0.1118,
                'D': 1.0,
                'zeta0': 0.05,
                'zeta': 0.2,
                'dtheta_PTO': 1e3, # threshold angular velocity for PTO
                'thickness': 0,
                'Ca': 0.5,
                'Cd': 0.5
            })
    elif witt_size == 'large':
        witt_params.update({
                'witt_size': 'large',
                'm': 110,
                'm_s': 7.31,
                'm_pto': 100,
                'm_b': 100,
                'l': 0.506,
                'D': 1.5,
                'zeta0': 0.05,
                'zeta': 0.2,
                'dtheta_PTO': 1e3, # threshold angular velocity for PTO
                'thickness': 0,
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
# initialise environmental parameters
env_params = { # define dictionary of parameters
    'g': 9.81,  # gravity
    'rho_w': 1025,  # water density
    'gamma_z': 0.3,
    'gamma_x': 0.3,
    'gamma_phi': 0.3,
    'a': np.array([0.5]),  # wave amplitudes
    'v': np.array([1]),  # wave frequencies
    'phis': np.array([0]),  # wave phases
}
env_params['k_w'] = (2 * np.pi * env_params['v']) ** 2 / env_params['g']
env_params['lambdas'] = 2 * np.pi / env_params['k_w']
print(f'wavelengths are: {env_params['lambdas']}') # print wavelengths
env_params['n_waves'] = len(env_params['a'])
print(f'number of waves: {env_params['n_waves']}')
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

# initialise mooring parameters
mooring_params = {
    'h': 5,  # depth of water (m)
    'alpha_m': np.pi/4,  # angle between mooring point on hull and vertical (rad)
    'X0': np.pi/4,  # angle between hypotenuse of mo
    'T0': 200,  # prestressed tension (N)
    'lambda_m0': 50,  # mooring stiffness when taut (N/m)
    'lambda_frac': 10,
    'slack_allowed': 1,
    'Kx': 200,
    'Kz': 100,
    'Kphi': 20,
    'Kxphi': 50
}
mooring_params['lambda_m'] = mooring_params['lambda_m0']*mooring_params['lambda_frac'] # mooring stiffness when taut, 10* of slack stiffness

#%% run simulations

# initial conditions [x, dxdt, z, dzdt, theta, dthetadt, phi, dphidt]
r0 = [0, 0, 0, 0, 3.14, 0, 0, 0]

# setup time parameters
t_start = 0
t_current = t_start
t_end = 200
t_step = 0.1
t_eval = np.arange(t_start, t_end, t_step)
t_span = (t_start, t_end)

## choose independent variable
amplitudes = [0.1, 0.3, 0.6, 0.9, 1.2, 1.5] # wave amplitudes
frequencies = [0.15, 0.3, 0.45] #, 0.6, 0.9, 1.2, 1.5, 1.8] # wave frequencies (Hz)
ballast_masses = [0, 10, 50, 100, 200, 300]
buoy_diams_small = [0.43, 0.5, 0.75, 1] #, 1.25, 1.5] #[1, 1.25, 1.5] #[0.395, 
buoy_diams_large = [1.25, 1.5, 1.75, 2, 2.25]
shell_masses = [5, 10, 25, 50, 75, 100]
pendulum_dampings = [0.1, 0.25, 0.5, 0.75, 1, 1.25]
witt_sizes = ['small','large']
added_mass_vals = [0, 0.25, 0.5, 0.75, 1]
Cd_vals = [0, 0.1, 0.25, 0.5, 0.75, 1]
n_moorings = [0, 1, 2]
depths = [5, 10, 20, 30, 40, 50, 75, 100]
mooring_alphas = np.array([1/8, 1/4, 3/8, 1/2])*np.pi
mooring_X0s = np.array([0, 1/8, 1/4, 3/8, 1/2])*np.pi
mooring_T0s = [50, 100, 150, 200, 250, 300]
lambda_m0s = [1000, 10000] # mooring stiffness when taut (N/m)
lambda_fracs = [2, 5, 10, 20]
slack_allowed = [1, 2, 3, 4, 5, 7.5, 10]

# run simulation, args = (WITT_function, mooring_type, n_mooring, t_span, r0, t_eval, method)
# pto_flag = True #False # set to True to include PTO damping
# print(f'PTO flag set to {pto_flag}, PTO damping will be included in simulation if True')
params = frequencies #buoy_diams_small
plot_labels = params
sols = []
Z0s = []
cairns_terms = []
PTO_terms = []
E_PTO = []
# mooring_params = update_params_mooring(mooring_params, T0=100)
mooring_params = update_params_mooring(mooring_params, X0=np.pi/3)
witt_params = update_params_witt(witt_params, m_b=200)
witt_params = update_params_witt(witt_params, dtheta_PTO=2)
# env_params = update_params_env(env_params, v=0.3)  # set wave parameters
print(f'Wave number is : {env_params['k_w']}')
event_flag = True
pto_calc = False # True
save_data = False
for param in params:
    ## update parameters
    # witt_params = update_params_witt(witt_params, Ca=param)
    env_params = update_params_env(env_params, v=param)
    # witt_params = update_witt_size(witt_size, witt_params)
    # mooring_params = update_params_mooring(mooring_params, lambda_m0=lambda_m0)
    
    # run simulation
    print(f'Running simulation for v = {param} Hz...')
    sol, Z0 = run_sim(WITT_function=nonlinear_2D_WITT, mooring_type='simplest_mooring', n_mooring=2, t_span=t_span, r0=r0, t_eval=t_eval, method='Radau', rtol=1e-3, atol=1e-6, env_params=env_params, witt_params=witt_params, mooring_params=mooring_params)
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
            Ftheta0 = -witt_params['c0'] * dthetadt  # natural damping force
            if np.abs(dthetadt) > witt_params['dtheta_PTO']:
                Ftheta = -witt_params['c'] * dthetadt + Ftheta0  # PTO damping force
                Power = np.abs((Ftheta - Ftheta0) * dthetadt)  # Power = Force * angular velocity
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
    plt.plot(sol.t, sol.y[var_id], label=label)
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
    plt.plot(sol.t, sol.y[var_id]-Z0s[i], label=label)
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
    plt.plot(sol.t, cairns_terms[i][0,:], label=label)
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
    plt.plot(sol.t, sol.y[var_id], label=label)
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
var_id = 8
for i,sol in enumerate(sols):
    label = plot_labels[i]
    # time_temp, angle_temp = wrap_angles(sol.t, sol.y[var_id])  # wrap angles to [-pi, pi]
    # plt.plot(time_temp, angle_temp, label=label)
    plt.plot(sol.t, sol.y[var_id], label=label)
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
    plt.plot(sol.t, sol.y[var_id], label=label)
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
    plt.plot(sol.t, sol.y[var_id], label=label)
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
    plt.plot(sol.t, cairns_terms[i][1,:], label=label)
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
    plt.plot(freqs[var_id, :], sols_amplitudes[i][var_id, :], label=label)
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
    plt.plot(freqs[var_id, :], sols_amplitudes[i][var_id, :], label=label)
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
    plt.plot(freqs[var_id, :], sols_amplitudes[i][var_id, :], label=label)
# plt.plot(sols_freqs[0][var_id, :], sols_amplitudes[0][var_id, :], label='n_mooring = 2')
# plt.plot(sols_freqs[1][var_id, :], sols_amplitudes[1][var_id, :], label='n_mooring = 1')
# plt.plot(sols_freqs[2][var_id, :], sols_amplitudes[2][var_id, :], label='n_mooring = 0')
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
    plt.plot(freqs[var_id, :], sols_amplitudes[i][var_id, :], label=label)
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
    output_folder = "02 Modelling/03 Summer Delivery/basic_mooring_varied_frequencies"  # set folder path
    os.makedirs(output_folder, exist_ok=True) # create the folder if it doesn't exist
    print(f'Saving data to {output_folder}')
    
    # want to save: sols, Z0, cairns_terms, PTO_terms
    sols_3d = np.array(sols) # convert sols to a 3D array
    np.savez(os.path.join(output_folder, "sim_outputs.npz"), sols_3d=sols_3d)
    np.savez(os.path.join(output_folder, "Z0s.npz"), Z0s=np.array(Z0s))
    np.savez(os.path.join(output_folder, "cairns_terms.npz"), cairns_terms=np.array(cairns_terms))
    np.savez(os.path.join(output_folder, "PTO_terms.npz"), PTO_terms=np.array(PTO_terms))
else:
    print('Simulation data not saved')
