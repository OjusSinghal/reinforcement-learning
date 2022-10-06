import numpy as np
import random
import matplotlib.pyplot as plt

actions = np.array([1, 2, 3, 4])
reward_means = np.array([0, 0, 0, 0.2])
reward_sigma = np.array([1, 0.7, 0.2, 0.5])

time_steps = 1000
epsilons = np.array([0.0, 0.2, 0.8, 1.0, 1.0])
decay_rates = np.array([0.0, 0.0, 0.0, 0.0, 0.05])

K = len(actions)
N = len(epsilons)

average_rewards = np.zeros(N)

for i in range(N):
    rewards = np.zeros(time_steps)
    estimates = np.zeros(K)
    action_counts = np.zeros(K)

    for step in range(time_steps):
        epsilon = epsilons[i] / (1 + step * decay_rates[i])
        if np.random.rand() < epsilon: action = random.choice(actions) - 1     
        else: action = np.argmax(estimates)

        rewards[step] = np.random.normal(reward_means[action], reward_sigma[action])
        action_counts[action] += 1
        estimates[action] += (rewards[step] - estimates[action]) / (action_counts[action])

    average_rewards[i] = rewards.mean()

    f = plt.figure(label="Epsilon = " + str(epsilon))
    f.set_figwidth(20)
    f.set_figheight(10)
    plt.xlabel("Time Step")
    plt.ylabel("Reward")
    plt.plot(rewards)

for i in range(len(epsilons)):
    print("Average Reward for epsilon = " + str(epsilons[i]) + " and epsilon-decay-rate = " + 
            str(decay_rates[i]) + " is: " + str(average_rewards[i]))

plt.show()