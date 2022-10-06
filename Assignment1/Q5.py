import numpy as np
import random
import matplotlib.pyplot as plt

K = 10
time_steps = 1000
episodes = 2000
ucb_c = 2
epsilon = 0.1
small_value = 1e-9

fig, ax = plt.subplots(1)
ax.set(xlabel="Time Step", ylabel="Average Reward")

fig.suptitle("Average performance of ucb vs epsilon-greedy")
fig.set_figwidth(20)
fig.set_figheight(10)

total_rewards = np.zeros([2, time_steps])

for episode in range(episodes):
    reward_means = np.random.randn(K)
    reward_sigma = np.full([K], 1)
    optimal_action = np.argmax(reward_means)

    estimates = np.zeros([2, K])
    action_counts = np.zeros([2, K])

    for step in range(time_steps):
        ## Epsilon-Greedy
        if np.random.rand() < epsilon: action = np.random.randint(0, K)     
        else: action = np.argmax(estimates[0])

        reward = np.random.normal(reward_means[action], reward_sigma[action])
        total_rewards[0][step] += reward
        estimates[0][action] += (reward - estimates[0][action]) / (action_counts[0][action] + 1)
        action_counts[0][action] += 1

        ## Upper Confidence Bound
        action = np.argmax(estimates[1] + ucb_c * np.power(np.log(step + 1) / (action_counts[1] + small_value), 0.5))

        reward = np.random.normal(reward_means[action], reward_sigma[action])
        total_rewards[1][step] += reward
        estimates[1][action] += (reward - estimates[1][action]) / (action_counts[1][action] + 1)
        action_counts[1][action] += 1    

average_rewards = total_rewards / episodes

ax.plot(average_rewards[0], label="Epsilon = " + str(epsilon))
ax.plot(average_rewards[1], label="UCB_C = " + str(ucb_c))
ax.legend(loc="lower right")
plt.show()