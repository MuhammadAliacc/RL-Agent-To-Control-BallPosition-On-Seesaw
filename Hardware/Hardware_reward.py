def calculate_reward(setpoint, position, beam_length, action):
    R1 = (1 - abs(setpoint - position))**2
    if position > 0:
        if action == 0:
            R2 = R1*1.2
        else:
            R2 = R1*0.8

    elif position < 0:
        if action == 2:
            R2 = R1*1.2
        else:
            R2 = R1*0.8

    else:
        R2 = R1*1.3

    return max(0, min(1, R2))