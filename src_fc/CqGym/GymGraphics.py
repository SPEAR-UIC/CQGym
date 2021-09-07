from matplotlib import pyplot as plt


class GymGraphics:

    def __init__(self, do_render=False, render_interval=1, render_pause=0.01):

        # Maintaining Variables for Rendering Visuals
        self.do_render = do_render
        self.render_interval = render_interval
        self.render_pause = render_pause
        if self.do_render:
            self.max_wait_times = []
            self.fig, ((self.node_graph, self.rewards_graph, self.max_wait_time_graph),
                       (self.que_wait_time_graph, self.que_req_time_graph, self.que_req_proc_graph)) = \
                plt.subplots(2, 3, figsize=(14, 5))

    def reset(self):
        if self.do_render:
            self.max_wait_times = []
            self.fig, ((self.node_graph, self.rewards_graph, self.max_wait_time_graph),
                       (self.que_wait_time_graph, self.que_req_time_graph, self.que_req_proc_graph)) =\
                plt.subplots(2, 3, figsize=(14, 5))
            self.fig.tight_layout(pad=3.0)

    @staticmethod
    def get_que_data_arrays(state):

        que_ids = state.wait_que
        que_wait_times = [state.current_time - state.job_info[idx]['submit'] for idx in que_ids]
        que_req_times = [state.job_info[idx]['reqTime'] for idx in que_ids]
        que_req_procs = [state.job_info[idx]['reqProc'] for idx in que_ids]

        return [str(idx) for idx in que_ids], que_wait_times, que_req_times, que_req_procs

    def visualize_data(self, iter, state, rewards):

        if self.do_render and state and iter % self.render_interval == 0:

            self.node_graph.clear()
            self.node_graph.bar(['Used Nodes', 'Idle Nodes'],
                                [state.total_nodes-state.idle_nodes,
                                 state.idle_nodes])
            self.node_graph.set_title('Used Nodes vs Idle Nodes')
            self.node_graph.set_ylim([0, 5000])

            self.rewards_graph.plot(rewards)
            self.rewards_graph.set_title('Rewards at each step')

            que_ids, que_wait_times, que_req_times, que_req_procs = self.get_que_data_arrays(state)

            self.que_wait_time_graph.clear()
            self.que_wait_time_graph.bar(que_ids, que_wait_times)
            self.que_wait_time_graph.set_title('Wait Time of Queued Jobs')

            self.max_wait_times.append(max(que_wait_times))
            self.max_wait_time_graph.plot(self.max_wait_times)
            self.max_wait_time_graph.set_title('Max wait time at each step')

            self.que_req_time_graph.clear()
            self.que_req_time_graph.bar(que_ids, que_req_times)
            self.que_req_time_graph.set_title('Req Time of Queued Jobs')

            self.que_req_proc_graph.clear()
            self.que_req_proc_graph.bar(que_ids, que_req_procs)
            self.que_req_proc_graph.set_title('Req Proc of Queued Jobs')

            plt.ion()
            plt.show()
            plt.pause(self.render_pause)
