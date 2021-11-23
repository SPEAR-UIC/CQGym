from CqGym.Gym import CqsimEnv
from Models.PPO import PPO
import numpy as np
import torch
import torch.optim as optim


def get_action_from_output_vector(output_vector, wait_queue_size, is_training):
    action_p = torch.softmax(
        output_vector[:wait_queue_size], dim=-1)
    action_p = np.array(action_p)
    action_p /= action_p.sum()
    if is_training:
        wait_queue_ind = np.random.choice(len(action_p), p=action_p)
    else:
        wait_queue_ind = np.argmax(action_p)
    return wait_queue_ind


def model_training(env, weights_file_name=None, is_training=False, output_file_name=None,
                   window_size=50, sys_size=0, learning_rate=0.1, gamma=0.99, batch_size=10, do_render=False, layer_size=[]):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    num_inputs = window_size * 2 + sys_size * 1
    ppo = PPO(env, num_inputs, window_size, std=0.0, window_size=window_size,
              learning_rate=learning_rate, gamma=gamma, batch_size=batch_size, layer_size=layer_size)

    if weights_file_name:
        ppo.load_using_model_name(weights_file_name)

    obs = env.get_state()
    done = False

    while not done:

        env.render()

        state = torch.FloatTensor(obs.feature_vector).to(device)

        probs, value = ppo.select_action(state)

        action_p = torch.softmax(probs.detach(), dim=-1)

        action = get_action_from_output_vector(
            probs.detach(), obs.wait_que_size, is_training)

        new_obs, done, reward = env.step(action)
        next_state = torch.FloatTensor(new_obs.feature_vector).to(device)

        ppo.remember(probs, value, reward, done, device,
                     action, state, next_state, action_p, obs)
        if is_training and not done:
            ppo.train()
        obs = new_obs

    if is_training and output_file_name:
        ppo.save_using_model_name(output_file_name)


def model_engine(module_list, module_debug, job_cols=0, window_size=0, sys_size=0,
                 is_training=False, weights_file=None, output_file=None, do_render=False, learning_rate=0.00001, reward_discount=0.99, batch_size=10, layer_size=[]):
    """
   Execute the CqSim Simulator using OpenAi based Gym Environment with Scheduling implemented using DeepRL Engine.

    :param module_list: CQSim Module :- List of attributes for loading CqSim Simulator
    :param module_debug: Debug Module :- Module to manage debugging CqSim run.
    :param job_cols: [int] :- No. of attributes to define a job.
    :param window_size: [int] :- Size of the input window for the DeepLearning (RL) Model.
    :param is_training: [boolean] :- If the weights trained need to be saved.
    :param weights_file: [str] :- Existing Weights file path.
    :param output_file: [str] :- File path if the where the new weights will be saved.
    :return: None
    """
    cqsim_gym = CqsimEnv(module_list, module_debug,
                         job_cols, window_size, do_render)
    model_training(cqsim_gym, window_size=window_size, sys_size=sys_size, is_training=is_training,
                   weights_file_name=weights_file, output_file_name=output_file, learning_rate=learning_rate, gamma=reward_discount, batch_size=batch_size, layer_size=layer_size)
