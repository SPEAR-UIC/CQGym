from CqGym.Gym import CqsimEnv
from Models.DQL import DQL
import tensorflow.compat.v1 as tf
import numpy as np
import time
tf.disable_v2_behavior()


def get_action_from_output_vector(output_vector):
    return np.argmax(output_vector)


def model_training(env, weights_file_name=None, is_training=False, output_file_name=None,
                   window_size=50, sys_size=0, learning_rate=0.1, gamma=0.99, batch_size=10, do_render=False, layer_size=[]):

    start = time.time()
    sess = tf.Session()
    tf.keras.backend.set_session(sess)
    dql = DQL(env, sess, window_size, learning_rate, gamma, batch_size, layer_size)

    if weights_file_name:
        dql.load_using_model_name(weights_file_name)

    obs = env.get_state()
    done = False

    while not done:

        env.render()
        output_vector = dql.act(obs.feature_vector)

        action = get_action_from_output_vector(output_vector)
        new_obs, done, reward = env.step(action)
        dql.remember(obs.feature_vector, output_vector, reward, new_obs.feature_vector)
        if is_training:
            dql.train()
        obs = new_obs

    if is_training and output_file_name:
        dql.save_using_model_name(output_file_name)
    
    print(sum(dql.reward_seq))


def model_engine(module_list, module_debug, job_cols=0, window_size=0, sys_size=0,
                 is_training=False, weights_file=None, output_file=None, do_render=False,
                 learning_rate=1e-5, reward_discount=0.99, batch_size=10, layer_size=[]):
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
    model_training(cqsim_gym, window_size=window_size, is_training=is_training,
                   weights_file_name=weights_file, output_file_name=output_file, sys_size=sys_size, learning_rate=learning_rate,
                   gamma=reward_discount, batch_size=batch_size, layer_size=layer_size)
