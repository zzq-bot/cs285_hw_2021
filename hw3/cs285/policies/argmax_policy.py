import numpy as np


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        ## TODO return the action that maxinmizes the Q-value 
        # at the current observation as the output
        batch_size = observation.shape[0]
        qa_values = self.critic.qa_values(observation)#numpy
        assert len(qa_values.shape) == 2#batch_size * action_dim
        assert qa_values.shape[0] == observation.shape[0]
        action = np.argmax(qa_values, axis=1)
        assert action==action.squeeze(), action.shape
        return action.squeeze()