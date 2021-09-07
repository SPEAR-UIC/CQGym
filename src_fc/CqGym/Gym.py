from CqSim.Cqsim_sim import Cqsim_sim
from gym import Env, spaces
import numpy as np
from CqGym.GymState import GymState
from CqGym.GymGraphics import GymGraphics
from copy import deepcopy

class CqsimEnv(Env):

    def __init__(self, module, debug=None, job_cols=0, window_size=0, do_render=False, render_interval=1, render_pause=0.01):
        Env.__init__(self)

        # Maintaining Variables for reset.
        self.simulator_module = module
        self.simulator_debug = debug

        # Initializing CQSim Backend
        self.simulator = Cqsim_sim(module, debug=debug)
        self.simulator.start()
        # Let Simulator load completely.
        self.simulator.pause_producer()

        GymState._job_cols_ = job_cols
        GymState._window_size_ = window_size
        self.gym_state = GymState()

        # Defining Action Space and Observation Space.
        self.action_space = spaces.Discrete(window_size)
        self.observation_space = spaces.Box(shape=(1, self.simulator.module['node'].get_tot() +
                                                   window_size * job_cols, 2),
                                            dtype=np.float32, low=0.0, high=1000000.0)

        # Define object for Graph Visualization:
        self.graphics = GymGraphics(do_render, render_interval, render_pause)
        self.rewards = []
        self.iter = 0

    def reset(self):
        """
        Reset the Gym Environment and the Simulator to a fresh start.
        :return: None
        """
        del self.simulator
        self.simulator = Cqsim_sim(deepcopy(self.simulator_module), debug=self.simulator_debug)
        self.simulator.start()
        # Let Simulator load completely.
        self.simulator.pause_producer()

        # Reinitialize Local variables
        self.gym_state = GymState()
        self.graphics.reset()
        self.rewards = []
        self.iter = 0

    def render(self, mode='human'):
        """
        :param mode: [str] :- No significance in the current version, only maintained to adhere to OpenAI-Gym standards.
        :return: None
        """
        # Show graphics at intervals.
        self.graphics.visualize_data(self.iter, self.gym_state, self.rewards)

    def get_state(self):
        """
        This function creates GymState Object for maintaining the current state of the Simulator.
        :return: [GymState]
        """
        self.gym_state = GymState()
        self.gym_state.define_state(self.simulator.currentTime,  # Current time in the simulator.
                                    self.simulator.simulator_wait_que_indices,  # Current Wait Queue in focus.
                                    self.simulator.module['job'].job_info(-1),  # All the JobInfo Dict.
                                    self.simulator.module['node'].nodeStruc,  # All the NodeStruct Dict.
                                    self.simulator.module['node'].get_idle())  # Number of Nodes available.
        return self.gym_state

    def step(self, action: int):
        """
        :param action: [int] :- Wait-Queue index of the selected Job.
                                Note - this is not Job Index.
        :return:
        gym_state: [GymState]   :- Contains all the information for the next state.
                                   gym_state.feature_vector stores Feature vector for the current state.
        done: [boolean]         :- True - If the simulation is complete.
        reward : [float]        :- reward for the current action.
        """
        self.iter += 1
        ind = action
        print("Wait Queue at Step Func - ", self.simulator.simulator_wait_que_indices)
        self.simulator.simulator_wait_que_indices = [self.simulator.simulator_wait_que_indices[ind]] + \
                                                     self.simulator.simulator_wait_que_indices[:ind] + \
                                                     self.simulator.simulator_wait_que_indices[ind + 1:]
        reward = self.gym_state.get_reward(self.simulator.simulator_wait_que_indices[0])

        # Maintaining data for GymGraphics
        self.rewards.append(reward)

        # Gym Paused, Running simulator.
        self.simulator.pause_producer()

        # Simulator executed with selected action. Retrieving new State.
        if self.simulator.is_simulation_complete:
            # Return an empty state if the Simulation is complete. Avoids NullPointer Exceptions.
            self.gym_state = GymState()
        else:
            self.get_state()

        return self.gym_state, self.simulator.is_simulation_complete, reward
