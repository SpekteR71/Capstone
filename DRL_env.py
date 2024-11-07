'''
Updated the code to work with only discrete values. Edit: This didnt work well. as each variable in state gets rounded to 0
Updated code to discretize only action output
Updated the reward function
'''


import pybullet as p
import pybullet_data
import time
import math
import numpy as np

class TwoWheelerEnv:
    def __init__(self):
        """Initialize the PyBullet simulation and load URDFs."""
        # Connect to PyBullet
        self.physics_client = p.connect(p.GUI)

        # Set gravity
        p.setGravity(0, 0, -9.81)           # we set gravity here right?
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(1./120.)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)                          # last 3 lines not needed. only first 3
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)           # just to make the gui easier to debug
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 1)    # including all thesse options hence there was no gravity after reset.hence dua lipa
        #p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 1)  # Shows collision shapes as wireframes


        # Load the plane
        plane_urdf_path = "road.urdf"
        try:
            self.plane_id = p.loadURDF(plane_urdf_path, basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1])
            print("Custom colored plane loaded successfully")
        except Exception as e:
            print(f"Error loading custom plane: {e}")

        # Load the two-wheeler
        self.base_height = 0.118  # Adjust this based on your model's base height
        urdf_path = "full_bike_v11_urdf/urdf/full_bike_v11_urdf.urdf"
        try:
            self.model_id = p.loadURDF(urdf_path, basePosition=[0, 0, self.base_height], baseOrientation=[0, 0, 0, 1])
            print("URDF loaded successfully")
        except Exception as e:
            print(f"Error loading URDF: {e}")
            self.model_id = None

        # Wait for 10 seconds before starting the simulation
        time.sleep(2)

        # Define joint indices
        self.momentum_wheel_index = 1
        self.steering_joint_index = 11
        self.front_wheel_index = 12
        self.driven_pulley_index = 13
        self.back_wheel_index = 14
        self.A = 0.5
        self.B = -0.5

    def reset(self):
        """Reset the environment to its initial state."""
        # Reset the simulation
        p.resetSimulation()

        # Set gravity again after resetting the simulation
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(1./120.)
        # Reload the plane and model URDFs without reconnecting to PyBullet
        plane_urdf_path = "road.urdf"
        try:
            self.plane_id = p.loadURDF(plane_urdf_path, basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1])
            print("Custom colored plane loaded successfully")
        except Exception as e:
            print(f"Error loading custom plane: {e}")

        # Load the two-wheeler
        urdf_path = "full_bike_v11_urdf/urdf/full_bike_v11_urdf.urdf"
        try:
            self.model_id = p.loadURDF(urdf_path, basePosition=[0, 0, self.base_height], baseOrientation=[0, 0, 0, 1])
            print("URDF loaded successfully")
        except Exception as e:
            print(f"Error loading URDF: {e}")
            self.model_id = None

        # Add a short delay to ensure the environment stabilizes before starting
        time.sleep(1)  # Adjust the sleep time as needed
        
        # Get the initial state after reset
        state = self.getData()

        # Return the state as a NumPy array
        return np.array(state, dtype=np.float32)


    def update_reward_factors(self, decrement=0.1):
        # Update the reward factors A and B
        if self.A > 0.6:
            self.A = max(0.6, self.A - decrement)
        if self.B < 0.4:
            self.B = min(0.4, self.B + decrement)

    def calculate_reward(self, state, next_state, destination_x, destination_y):
        """Calculate the navigation and balance rewards based on current and next states."""
        
        # Unpack current state
        x, y = state[0], state[1]
        roll = state[7]
        
        # Unpack next state
        next_x, next_y = next_state[0], next_state[1]
        next_roll = next_state[7]
        
        # Calculate the distance to the target from current and next state
        current_distance_to_target = np.linalg.norm([destination_x - x, destination_y - y])
        next_distance_to_target = np.linalg.norm([destination_x - next_x, destination_y - next_y])
        print(current_distance_to_target, next_distance_to_target)
        # Navigation reward: difference in distance to the target
        if abs(current_distance_to_target) > abs(next_distance_to_target):
            navigation_reward =  self.A * (abs(current_distance_to_target) - abs(next_distance_to_target))
        else:
            navigation_reward =  self.B * (abs(next_distance_to_target) - abs(current_distance_to_target))
        
        # Balance reward: based on the change in roll angle (tilt)
        if abs(roll) > abs(next_roll):
            balance_reward = self.A * (abs(roll) - abs(next_roll))
        else:
            balance_reward = self.B * (abs(next_roll) - abs(roll))

        # Combine the two rewards
        total_reward = 2 * navigation_reward + 5 * balance_reward
        
        return total_reward

    def step(self, state, action, destination_x, destination_y):
        """Perform a single step in the simulation using the given action."""

        momentum_wheel_input, back_wheel_input, steering_input = action
        
        #momentum_wheel_input = int(round(action[0]))
        #back_wheel_input = int(round(action[1]))
        #steering_input = int(round(action[2]))
        
        #print(state, momentum_wheel_input, back_wheel_input, steering_input)
        # Apply the action to the simulation
        p.setJointMotorControl2(
            self.model_id,
            self.momentum_wheel_index,
            p.VELOCITY_CONTROL,
            targetVelocity=momentum_wheel_input,
            force=0.255     # torque
        )

        p.setJointMotorControl2(
            self.model_id, 
            self.driven_pulley_index, 
            p.VELOCITY_CONTROL, 
            targetVelocity=-back_wheel_input, 
            force=0.4
        )

        p.setJointMotorControl2(
            self.model_id,
            self.back_wheel_index,
            p.VELOCITY_CONTROL,
            targetVelocity=back_wheel_input,
            force=0.4
        )

        p.setJointMotorControl2(
            self.model_id,
            self.front_wheel_index,
            p.VELOCITY_CONTROL,
            targetVelocity=back_wheel_input,
            force=0.4
        )


        p.setJointMotorControl2(
            self.model_id,
            self.steering_joint_index,
            p.POSITION_CONTROL,
            targetPosition=steering_input,
            force=0.08
        )

        # Step the simulation
        p.stepSimulation()
        next_state = self.getData()
        
        # Check if the destination is reached
        x, y = next_state[0], next_state[1]
        destination_reached = self._check_destination_reached(x, y, destination_x, destination_y)

        # Check if two wheeler has rolled over
        roll, pitch = next_state[7], next_state[8]
        tipped_over = abs(roll) > math.radians(45) or abs(pitch) > math.radians(45)

        # Define reward and done condition
        reward = self.calculate_reward(state, next_state, destination_x, destination_y)
        done = destination_reached or tipped_over
        print(reward)
        if destination_reached:
            reward = 2
            return np.array(next_state, dtype=np.float32), reward, done
        if tipped_over:
            reward = -2
            return np.array(next_state, dtype=np.float32), reward, done
        # Return next_state, reward, and done as a tuple
        return np.array(next_state, dtype=np.float32), reward, done

    def _check_destination_reached(self, x, y, destination_x, destination_y):
        """Check if the destination has been reached."""
        distance_threshold = 0.5
        distance = np.sqrt((x - destination_x) ** 2 + (y - destination_y) ** 2)
        return distance <= distance_threshold

    def getData(self):
        """Simulate sensor reading data from the environment with integer values."""
        # Get position and orientation
        position, orientation = p.getBasePositionAndOrientation(self.model_id)
        linear_velocity, angular_velocity = p.getBaseVelocity(self.model_id)
        
        # Convert orientation from quaternion to Euler angles
        roll, pitch, yaw = p.getEulerFromQuaternion(orientation)
        
        # Get x and y coordinates from position
        x, y, z = position

        # Flatten linear and angular velocity tuples
        linear_velocity = list(linear_velocity)
        angular_velocity = list(angular_velocity)

        # Example scaling factor
        state = [x, y] + linear_velocity + angular_velocity + [roll, pitch, yaw]
        #scaled_state = state * 1000
        #discrete_state = np.floor(scaled_state).astype(int)
        # Return data as a flat list of integer values
        return state

