#!pip install gymnasium
import gymnasium as gym
import numpy as np

def value_iteration(env, gamma=0.99, theta=1e-6):
    """Perform value iteration algorithm"""
    value_table = np.zeros(env.observation_space.n)
    num_actions = env.action_space.n
    P = env.unwrapped.P
    
    while True:
        delta = 0
        for state in range(env.observation_space.n):
            v = value_table[state]
            max_value = float('-inf')
            
            for action in range(num_actions):
                action_value = 0
                for prob, next_state, reward, done in P[state][action]:
                    action_value += prob * (reward + gamma * value_table[next_state])
                max_value = max(max_value, action_value)
            
            value_table[state] = max_value
            delta = max(delta, abs(v - value_table[state]))
        
        if delta < theta:
            break
    
    return value_table

def extract_policy(env, value_table, gamma=0.99):
    """Extract policy from value table"""
    policy = np.zeros(env.observation_space.n, dtype=int)
    P = env.unwrapped.P
    
    for state in range(env.observation_space.n):
        action_values = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            action_value = 0
            for prob, next_state, reward, done in P[state][action]:
                action_value += prob * (reward + gamma * value_table[next_state])
            action_values[action] = action_value
        policy[state] = np.argmax(action_values)
    
    return policy

def run_simulation(env, policy):
    """Run simulation with the given policy"""
    state = env.reset()[0]
    done = False
    total_reward = 0

    while not done:
        action = policy[state]
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        state = next_state
        env.render()
    
    return total_reward

def main():
    # Create environment
    env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=False)
    env.reset()
    
    # Run value iteration
    value_table = value_iteration(env)
    
    # Extract policy
    optimal_policy = extract_policy(env, value_table)
    
    # Display results
    print("Optimal Policy:")
    print(optimal_policy.reshape((4,4)))
    
    # Run simulation
    total_reward = run_simulation(env, optimal_policy)
    print(f"Total reward using optimal policy: {total_reward}")
    
    env.close()

if __name__ == "__main__":
    main()