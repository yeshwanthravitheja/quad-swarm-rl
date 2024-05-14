def crazyflie_params():
    ## See: Ref[2] for details
    ## Geometric parameters for Inertia and the model
    geom_params = {}
    geom_params["body"] = {"l": 0.03, "w": 0.03, "h": 0.004, "m": 0.005}
    geom_params["payload"] = {"l": 0.035, "w": 0.02, "h": 0.008, "m": 0.027}
    geom_params["arms"] = {"l": 0.022, "w": 0.005, "h": 0.005, "m": 0.001}
    geom_params["motors"] = {"h": 0.02, "r": 0.0035, "m": 0.0015}
    geom_params["propellers"] = {"h": 0.002, "r": 0.022, "m": 0.00075}

    geom_params["motor_pos"] = {"xyz": [0.065 / 2, 0.065 / 2, 0.]}
    geom_params["arms_pos"] = {"angle": 45., "z": 0.}
    geom_params["payload_pos"] = {"xy": [0., 0.], "z_sign": 1}
    # z_sing corresponds to location (+1 - on top of the body, -1 - on the bottom of the body)

    ## Damping parameters
    # damp_params = {"vel": 0.001, "omega_quadratic": 0.015}
    damp_params = {"vel": 0.0, "omega_quadratic": 0.0}

    ## Noise parameters
    noise_params = {}
    noise_params["thrust_noise_ratio"] = 0.05

    ## Motor parameters
    motor_params = {"thrust_to_weight": 1.9,  #2.18
                    "assymetry": [1.0, 1.0, 1.0, 1.0],
                    "torque_to_thrust": 0.006,  #0.005964552
                    "linearity": 1.0,  #0.424 for CrazyFlie w/o correction in firmware (See [2])
                    "C_drag": 0.000,  # 3052 * 9.1785e-07  #3052 * 8.06428e-05, # 0.246
                    "C_roll": 0.000,  #3052 * 0.000001 # 0.0003
                    "damp_time_up": 0.15,
                    #0.15, #0.15 - See: [4] for details on motor damping. Note: these are rotational velocity damp params.
                    "damp_time_down": 0.15  #2.0, #2.0
                    }

    ## Summarizing
    params = {
        "geom": geom_params,
        "damp": damp_params,
        "noise": noise_params,
        "motor": motor_params
    }
    return params
