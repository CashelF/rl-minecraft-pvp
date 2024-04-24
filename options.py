import math
import time

NO_OP_ACTION = 0

def look_at_zombie(observation, env):
    # while True:
        # env.step(NO_OP_ACTION)
    
    max_attempts = 100
    attempts = 0

    while attempts < max_attempts:
        if not observation or 'XPos' not in observation or 'ZPos' not in observation or 'Yaw' not in observation:
            print("Missing necessary positional information in observation.")
            return
        # Retrieve the first entity in the observations assumed to be a zombie
        entities = observation.get('entities', [])
        zombie_observation = next((ent for ent in entities if ent.get('name') == 'Zombie'), None)

        if not zombie_observation:
            env.send_command("turn 1")
            # time.sleep(0.001)  # Give some time for the command to process and new observation to be generated
            # Hypothetical function to fetch new observations
            _, _, _, info = env.step(NO_OP_ACTION)
            observation = info['observation']
            attempts += 1
            continue
        
        # Calculate the direction to the zombie
        dx = zombie_observation['x'] - observation['XPos']
        dz = zombie_observation['z'] - observation['ZPos']
        target_yaw = math.atan2(dz, dx) * 180 / math.pi

        # Calculate the difference in yaw
        yaw_diff = target_yaw - observation['Yaw']
        if yaw_diff > 180:
            yaw_diff -= 360
        elif yaw_diff < -180:
            yaw_diff += 360

        # If yaw difference is negligible, assume the zombie is in view
        if abs(yaw_diff) < 5:  # Adjust tolerance as needed
            return

        # Send turn command based on the yaw difference
        turn_command = "turn " + str(max(-1, min(1, yaw_diff / 180)))
        try:
            env.send_command(turn_command)
            # time.sleep(0.001)  # Simulate waiting for the command to take effect
            # Update observation to see if the zombie is now in view
            _, _, _, info = env.step(NO_OP_ACTION)
            observation = info['observation']
        except Exception as e:
            print("Error sending command:", e)
        
        attempts += 1
