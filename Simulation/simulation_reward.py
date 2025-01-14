
def calculate_reward(setpoint, position, beam_length, action, velocity, Use_Reward_function_advance):
    R1 = (1.0 - abs(setpoint - position) / beam_length) ** 2

    if Use_Reward_function_advance==True:
        #print(' Using reward function advance ')
        if position > 0:
            if action == 0:
                R2 = R1*1.3
            else:
                R2 = R1*0.8
        elif position < 0:
            if action == 2:
                R2 = R1*1.3
            else:
                R2 = R1*0.8
        else:
            R2 = R1*1.3

        total_reward = R2 

        return max(0, min(1, total_reward))
    else:
        return max(0, min(1, R1))