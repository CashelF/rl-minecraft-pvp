import math

def look_at_zombie(observation, env):
    if not observation or not 'XPos' in observation or not 'ZPos' in observation or not 'Yaw' in observation:
        for i in range(100):
            env.send_command('turn 1')
        return
    zombie_observation = observation.get('entities', [{}])[0]
    
    if not zombie_observation:
        env.send_command("turn 1")
        return
    
    # Calculate the direction to the zombie
    dx = zombie_observation['x'] - observation['XPos']
    dz = zombie_observation['z'] - observation['ZPos']
    target_yaw = math.atan2(dz, dx) * 180 / math.pi

    # Calculate the difference in yaw and adjust accordingly
    yaw_diff = target_yaw - observation['Yaw']
    if yaw_diff > 180:
        yaw_diff -= 360
    elif yaw_diff < -180:
        yaw_diff += 360

    command = "turn "
    if yaw_diff > 0:
        command += str(min(1, yaw_diff / 180))
    else:
        command += str(max(-1, yaw_diff / 180))

    # Send the command to the environment
    env.send_command('command')
    # You might want to add a delay or perform this in steps, checking the agent's updated state each time
