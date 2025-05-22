# CART POLE BALANCING

## AIM
To implement and evaluate a Monte Carlo control algorithm for optimizing state-action values in a gym environment, using discretized states and policy updates.

## PROBLEM STATEMENT
Explain the problem statement.

## MONTE CARLO CONTROL ALGORITHM FOR CART POLE BALANCING
### Step 1:
Environment Setup: Import gym and initialize the environment (e.g., CartPole). Set hyperparameters for bins, learning rates, and exploration rates.
### Step 2:
Discretization: Divide the continuous state space into discrete bins to facilitate learning.
### Step 3:
Policy Initialization: Initialize Q-values (state-action values) and define an epsilon-greedy policy for action selection.
### Step 4:
Episode Generation: Generate trajectories by interacting with the environment using the policy. Each trajectory consists of state, action, reward, and next state information.
### Step 5:
Return Calculation: Compute the cumulative discounted return 
### Step 6:
Policy Improvement: Update the policy to select actions that maximize the updated Q-values for each state.
### Step 7:
Convergence: Repeat the process until Q-values stabilize or a predefined number of episodes is reached.

## MONTE CARLO CONTROL FUNCTION

### Importing necessary Modules
```

import gymnasium as gym
import numpy as np
from itertools import count
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
```

```
g_bins = 10
Q_track = 0
Q = 0
```
```
def create_bins(n_bins=g_bins, n_dim=4):

    bins = [
        np.linspace(-4.8, 4.8, n_bins),
        np.linspace(-4, 4, n_bins),
        np.linspace(-0.418, 0.418, n_bins),
        np.linspace(-4, 4, n_bins)
    ]

    return bins

def discretize_state(observation, bins):

    binned_state = []

    for i in range(len(observation)):
        d = np.digitize(observation[i], bins[i])
        binned_state.append( d - 1)

    return tuple(binned_state)
```
### Decay schedule Funtion
```
def decay_schedule(init_value, min_value, decay_ratio, max_steps, log_start=1, log_base=10):
    """
    Decays a value from an initial value to a minimum value using a logarithmic schedule.

    Args:
        init_value: The initial value.
        min_value: The minimum value.
        decay_ratio: The decay ratio.
        max_steps: The maximum number of steps.
        log_start: The starting point for the logarithmic decay.
        log_base: The base of the logarithm.

    Returns:
        A NumPy array of decayed values.
    """
    decay_steps = int(max_steps * decay_ratio)
    rem_steps = max_steps - decay_steps  # changed rem_steps calculation

    # Check if decay_steps is zero and handle it to prevent empty array
    if decay_steps == 0:
        return np.full(max_steps, init_value)  # Return an array of init_value

    values = np.logspace(
        log_start, 0, decay_steps,
        base=log_base, endpoint=True)[::-1]
    values = (values - values.min()) / (values.max() - values.min())
    values = (init_value - min_value) * values + min_value
    values = np.pad(values, (0, rem_steps), 'edge')

    return values
```
### Generate Trajectory Function
```
def generate_trajectory(
    select_action, Q, epsilon,
    env, max_steps=200):
    done, trajectory = False, []
    bins = create_bins(g_bins)

    observation,_ = env.reset()
    state = discretize_state(observation, bins)

    for t in count():
        action = select_action(state, Q, epsilon)
        observation, reward, done, _, _ = env.step(action)
        next_state = discretize_state(observation, bins)
        if not done:
            if t >= max_steps-1:
                break
            experience = (state, action,
                    reward, next_state, done)
            trajectory.append(experience)
        else:
            experience = (state, action,
                    -100, next_state, done)
            trajectory.append(experience)
            #time.sleep(2)
            break
        state = next_state

    return np.array(trajectory, dtype=object)
```
### Monte-Caro Funtion
```
def mc_control (env,n_bins=g_bins, gamma = 1.0,
                init_alpha = 0.5,min_alpha = 0.01, alpha_decay_ratio = 0.5,
                init_epsilon = 1.0, min_epsilon = 0.1, epsilon_decay_ratio = 0.9,
                n_episodes = 3000, max_steps = 200, first_visit = True, init_Q=None):

    nA = env.action_space.n
    discounts = np.logspace(0, max_steps,
                            num = max_steps, base = gamma,
                            endpoint = False)
    alphas = decay_schedule(init_alpha, min_alpha,
                            0.9999, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon,
                            0.99, n_episodes)
    pi_track = []
    global Q_track
    global Q


    if init_Q is None:
        Q = np.zeros([n_bins]*env.observation_space.shape[0] + [env.action_space.n],dtype =np.float64)
    else:
        Q = init_Q

    n_elements = Q.size
    n_nonzero_elements = 0

    Q_track = np.zeros([n_episodes] + [n_bins]*env.observation_space.shape[0] + [env.action_space.n],dtype =np.float64)
    select_action = lambda state, Q, epsilon: np.argmax(Q[tuple(state)]) if np.random.random() > epsilon else np.random.randint(len(Q[tuple(state)]))

    progress_bar = tqdm(range(n_episodes), leave=False)
    steps_balanced_total = 1
    mean_steps_balanced = 0
    for e in progress_bar:
        trajectory = generate_trajectory(select_action, Q, epsilons[e],
                                    env, max_steps)

        steps_balanced_total = steps_balanced_total + len(trajectory)
        mean_steps_balanced = 0

        visited = np.zeros([n_bins]*env.observation_space.shape[0] + [env.action_space.n],dtype =np.float64)
        for t, (state, action, reward, _, _) in enumerate(trajectory):
            #if visited[tuple(state)][action] and first_visit:
            #    continue
            visited[tuple(state)][action] = True
            n_steps = len(trajectory[t:])
            G = np.sum(discounts[:n_steps]*trajectory[t:, 2])
            Q[tuple(state)][action] = Q[tuple(state)][action]+alphas[e]*(G - Q[tuple(state)][action])
        Q_track[e] = Q
        n_nonzero_elements = np.count_nonzero(Q)
        pi_track.append(np.argmax(Q, axis=env.observation_space.shape[0]))
        if e != 0:
            mean_steps_balanced = steps_balanced_total/e
        #progress_bar.set_postfix(episode=e, Epsilon=epsilons[e], Steps=f"{len(trajectory)}" ,MeanStepsBalanced=f"{mean_steps_balanced:.2f}", NonZeroValues="{0}/{1}".format(n_nonzero_elements,n_elements))
        progress_bar.set_postfix(episode=e, Epsilon=epsilons[e], StepsBalanced=f"{len(trajectory)}" ,MeanStepsBalanced=f"{mean_steps_balanced:.2f}")

    print("mean_steps_balanced={0},steps_balanced_total={1}".format(mean_steps_balanced,steps_balanced_total))
    V = np.max(Q, axis=env.observation_space.shape[0])
    pi = lambda s:{s:a for s, a in enumerate(np.argmax(Q, axis=env.observation_space.shape[0]))}[s]

    return Q, V, pi
```
```
env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset(seed=42)
```
```
# To run the MC control without using the previous Q values
optimal_Q, optimal_V, optimal_pi = mc_control (env,n_episodes=200)
```
```
#To save the action value function
np.save("state_action_values.npy", Q)
```
```
#To load the action value function
Q = np.load("state_action_values.npy")
```
```
observation, info = env.reset(seed=42)
observation, reward, done, _, _ = env.step(0)
print(done)
env.action_space.n
```
```
# To run the MC control using the previous Q values and default parameters
optimal_Q, optimal_V, optimal_pi = mc_control (env,n_episodes=200,
                                    init_alpha = 0.5,min_alpha = 0.01, alpha_decay_ratio = 0.5,
                                    init_epsilon = 1.0, min_epsilon = 0.1, epsilon_decay_ratio = 0.9,
`                                    max_steps=500, init_Q=Q)
```
```
# To run the MC control using the previous Q values and modified parameters
optimal_Q, optimal_V, optimal_pi = mc_control (env,n_episodes=500,
                                    init_alpha = 0.01,min_alpha = 0.005, alpha_decay_ratio = 0.5,
                                    init_epsilon = 0.1 , min_epsilon = 0.08, epsilon_decay_ratio = 0.9,
                                    max_steps=500, init_Q=Q)
```
## OUTPUT:

#### To run the MC control using the previous Q values and default paramete
![image](https://github.com/user-attachments/assets/980f543e-12f6-4c4f-8532-68eecc0236f1)


 
 #### To run the MC control using the previous Q values and modified parameters 
 ![image](https://github.com/user-attachments/assets/f7ca8c12-5aaf-433c-80c7-214e6ff2f490)





## RESULT:
The implemented algorithm successfully learned an optimal policy for balancing the pole in the environment, evidenced by an increase in mean steps balanced across episodes. The final policy maximized the expected reward for given states
