import numpy as np
import random
import matplotlib.pyplot as plt

K = 10
time_steps = 1000
episodes = 2000

epsilons = np.array([0.0, 0.01, 0.1])

N = len(epsilons)

fig, ax = plt.subplots(2)

ax.flat[0].set(xlabel="Time Step", ylabel="Average Reward")
ax.flat[1].set(xlabel="Time Step", ylabel="% Optimal Action")


fig.suptitle("Average performance of epsilon-greedy")
fig.set_figwidth(20)
fig.set_figheight(10)


total_rewards = np.zeros([N, time_steps])
total_optimal_actions = np.zeros([N, time_steps])

for episode in range(episodes):
    reward_means = np.random.randn(K)
    reward_sigma = np.full([K], 4)
    optimal_action = np.argmax(reward_means)

    estimates = np.zeros([N, K])
    action_counts = np.zeros([N, K])

    for model_num in range(N):
        epsilon = epsilons[model_num]

        for step in range(time_steps):
            if np.random.rand() < epsilon: action = np.random.randint(0, K)     
            else: action = np.argmax(estimates[model_num])

            reward = np.random.normal(reward_means[action], reward_sigma[action])
            total_rewards[model_num][step] += reward
            estimates[model_num][action] += (reward - estimates[model_num][action]) / (action_counts[model_num][action] + 1)
            action_counts[model_num][action] += 1

            if action == optimal_action:
                total_optimal_actions[model_num][step] += 1


average_rewards = total_rewards / episodes
perc_optimal_actions = total_optimal_actions * 100.0 / episodes

for model_num in range(N):
    ax[0].plot(average_rewards[model_num], label="Epsilon = " + str(epsilons[model_num]))
    ax[1].plot(perc_optimal_actions[model_num], label="Epsilon = " + str(epsilons[model_num]))


for axis in ax:
    axis.legend(loc="lower right")

plt.show()