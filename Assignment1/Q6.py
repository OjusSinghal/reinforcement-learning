import numpy as np
import random
import matplotlib.pyplot as plt

def softmax(vals):
    exps = np.exp(vals)
    summs = exps.sum(axis=2).reshape([exps.shape[0], exps.shape[1], 1])
    return exps / summs

arm_count = 10
time_steps = 1000
episodes = 2000
                                                                             
alphas = np.full([arm_count, 4], [0.4, 0.1, 0.4, 0.1]).T
## with baseline, without baseline

model_count = alphas.shape[0] ## 4
average_rewards = np.zeros([episodes, model_count])
action_preferences = np.full([episodes, model_count, arm_count], 1.0)

reward_means = np.random.normal(4.0, size=[episodes, arm_count])
optimal_actions = np.argmax(reward_means, axis=1)
action_counts = np.zeros([model_count, time_steps])


for step in range(time_steps):
    pmfs = softmax(action_preferences)

    for episode in range(episodes):
        for model_num in range(model_count):

            action = np.random.choice(arm_count, 1, p=pmfs[episode][model_num])
            reward = np.random.normal(reward_means[episode][action], 1.0)
            indicator = np.zeros(arm_count, dtype='f')
            indicator[action] = 1

            baseline = np.full([arm_count], average_rewards[episode][model_num])
            if model_num > 1: baseline = np.zeros(arm_count)

            update_pref = alphas[model_num] * (np.full([arm_count], reward) - baseline) * (indicator - pmfs[episode][model_num])
            action_preferences[episode][model_num] += update_pref

            average_rewards[episode][model_num] += (reward - average_rewards[episode][model_num]) / (step + 1)
            if action == optimal_actions[episode]:
                action_counts[model_num][step] += 1

perc_optimal_actions = (action_counts * 100.0 / episodes)

fig, ax = plt.subplots(1)
ax.set(xlabel="Time Step", ylabel="% Optimal Action")

fig.suptitle("Average performance of the Gradient Bandit Algorithm")
fig.set_figwidth(20)
fig.set_figheight(10)

for i in range(model_count):
    if i < 2: ax.plot(perc_optimal_actions[i], label="alpha = " + str(alphas[i][0]) + ", with baseline")
    else: ax.plot(perc_optimal_actions[i], label="alpha = " + str(alphas[i][0]) + ", without baseline")

ax.legend(loc="lower right")
plt.show()
