from threading import Condition
import time


class Pause:

    def __init__(self):
        """
        Class Pause to implement a Producer-Consumer Approach.
        2 Conditional variables are initialized - for each Prod and Cons.
        Conditional Variables implementation in Threads - https://docs.python.org/3/library/threading.html
        """
        self.prod_cv = Condition()
        self.cons_cv = Condition()
        self.initial_check = True

    def pause_producer(self):
        """
        First resume(notify) Consumer and then pause Producer.
        """
        with self.cons_cv:              # Implementing Locking for Conditional Variable.
            self.cons_cv.notifyAll()    # To notify all the Paused Consumer Threads to resume.
            # Since only 1 Consumer thread, can also use .notify()

        with self.prod_cv:
            self.prod_cv.wait()         # Pause the Producer until it gets a notification.

    def pause_consumer(self):
        """
        First resume(notify) Consumer and then pause Producer.
        """

        # *****
        # Maintained a special variable for the initial situation specific to CqSim-Gym.Env
        # *****
        self.initial_check = False

        with self.prod_cv:              # Implementing Locking for Conditional Variable.
            self.prod_cv.notifyAll()    # To notify that all Paused Producer Threads to resume.
        with self.cons_cv:
            self.cons_cv.wait()         # Pause the Consumer until it gets a notification.

    def is_producer_paused(self):
        """
        At the initialization of the CqSim-Gym.Env , the Env needs to provide the Current State of the Env.
        At this point we only need to wait until CqSim_Simulator(Consumer) has completely initialised the initial State.
        This function does not send any notification to the Conditional Variables.

        Note: This function still uses "While" loop, However it is only used ONCE in the CqSim-Gym.Env lifecycle.
              Hence does not add to overhead.
        """
        while self.initial_check:
            time.sleep(0.001)

    def release_all(self):
        """
        Once the Consumer(CqSim) and the Producer(Gym.Env) are completed i.e. all the jobs are loaded and assigned,
        both the threads notified and released to run and finish independently.
        """
        with self.prod_cv:
            self.prod_cv.notifyAll()
        with self.cons_cv:
            self.cons_cv.notifyAll()
