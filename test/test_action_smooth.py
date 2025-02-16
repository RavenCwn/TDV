import numpy as np


def action_smooth(all_time_actions, t):
    actions_for_curr_step = all_time_actions[:, t]
    actions_populated = (np.sum(actions_for_curr_step, axis=1) != 0)  # 所有的action都非0的时候才算是有效? 改成总的和非零才有用
    
    print("actions_for_curr_step: ", actions_for_curr_step.shape)
    print("actions_populated: ", actions_populated.shape)
    
    actions_for_curr_step = actions_for_curr_step[actions_populated]
    k = 0.01
    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
    exp_weights = exp_weights / np.sum(exp_weights)
    exp_weights = exp_weights[:, np.newaxis]
    actions = np.sum(actions_for_curr_step * exp_weights, axis=0)

    return actions


if __name__ == "__main__":
    steps = 200
    action_chunk = 15
    action_dim = 7
    
    all_time_actions = np.zeros([steps, steps+action_chunk, action_dim])
    
    for t in range(steps):
        actions = action_smooth(all_time_actions, t)
        print(actions.shape)