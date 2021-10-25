from CqGym.Gym import CqsimEnv
import numpy as np


def model_training(env, do_render=False):

    obs = env.get_state()
    done = False

    while not done:
        env.render()
        action = -1
        early_submit = float('Inf')
        for i, v in enumerate(obs.wait_job):
            if v['submit'] < early_submit:
                action = i
                early_submit = v['submit']
        new_obs, done, reward = env.step(action)


def model_engine(module_list, module_debug, job_cols=0, window_size=0, sys_size=0, do_render=False):
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
    model_training(cqsim_gym)
