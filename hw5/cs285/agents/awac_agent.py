from collections import OrderedDict

from cs285.critics.dqn_critic import DQNCritic
from cs285.critics.cql_critic import CQLCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *
from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.argmax_policy import ArgMaxPolicy
from cs285.infrastructure.dqn_utils import MemoryOptimizedReplayBuffer
from cs285.exploration.rnd_model import RNDModel
from .dqn_agent import DQNAgent
from cs285.policies.MLP_policy import MLPPolicyAWAC
import numpy as np
import torch


class AWACAgent(DQNAgent):
    def __init__(self, env, agent_params, normalize_rnd=True, rnd_gamma=0.99):
        super(AWACAgent, self).__init__(env, agent_params)
        
        self.replay_buffer = MemoryOptimizedReplayBuffer(100000, 1, float_obs=True)
        self.num_exploration_steps = agent_params['num_exploration_steps']
        self.offline_exploitation = agent_params['offline_exploitation']

        self.exploitation_critic = DQNCritic(agent_params, self.optimizer_spec)
        self.exploration_critic = DQNCritic(agent_params, self.optimizer_spec)
        
        self.exploration_model = RNDModel(agent_params, self.optimizer_spec)
        self.explore_weight_schedule = agent_params['explore_weight_schedule']
        self.exploit_weight_schedule = agent_params['exploit_weight_schedule']
        
        self.use_boltzmann = agent_params['use_boltzmann']
        self.actor = ArgMaxPolicy(self.exploitation_critic)
        self.eval_policy = self.awac_actor = MLPPolicyAWAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
            self.agent_params['awac_lambda'],
        )

        self.exploit_rew_shift = agent_params['exploit_rew_shift']
        self.exploit_rew_scale = agent_params['exploit_rew_scale']
        self.eps = agent_params['eps']

        self.running_rnd_rew_std = 1
        self.normalize_rnd = normalize_rnd
        self.rnd_gamma = rnd_gamma

    def get_qvals(self, critic, obs, action):
        # get q-value for a given critic, obs, and action
        if isinstance(obs, torch.Tensor):
            obs = ptu.to_numpy(obs)
        if isinstance(action, torch.Tensor):
            action = ptu.to_numpy(action)
        if len(obs.shape)==1:
            qa_value= critic.qa_values(obs)
            assert qa_value.size == critic.ac_dim, f"{qa_value.shape}"
            q_value = qa_value[action]
        else:
            batch_size = obs.shape[0]
            qa_values = critic.qa_values(obs)
            if action.size == 1:
                action = np.repeat(action, batch_size)
            assert qa_values.shape == (batch_size, critic.ac_dim) and action.size == batch_size
            q_value = np.choose(action, qa_values.T)#每一行中以action为索引选取元素
            assert q_value.shape == (batch_size, )
        return q_value

    def estimate_advantage(self, ob_no, ac_na, re_n, next_ob_no, terminal_n, n_actions=10):
        # TODO convert to torch tensors
        assert len(ob_no.shape) == 2
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na).to(torch.long)
        next_ob_no = ptu.from_numpy(next_ob_no)
        re_n = ptu.from_numpy(re_n)
        terminal_n = ptu.from_numpy(terminal_n)
        batch_size = ob_no.shape[0]
        vals = []
        # TODO Calculate Value Function Estimate given current observation
        # You may find it helpful to utilze get_qvals defined above
        #先evaluate Q(s,a) for all action V(s) = \sum \pi(a|s) * Q(s,a)
        #set a critic for getting q_val
        critic = self.actor.critic
        #critic = self.exploitation_critic
        if self.agent_params['discrete']:
            action_distribution = self.awac_actor(ob_no)
            for i in range(self.agent_params['ac_dim']):
                ac_i = np.repeat([i], batch_size)
                q_sa = self.get_qvals(critic, ob_no, ac_i)
                pi_sa = ptu.to_numpy(action_distribution.log_prob(ptu.from_numpy(ac_i))).squeeze()
                assert pi_sa.shape == q_sa.shape == (batch_size, )
                vals.append(q_sa * pi_sa)
        else:
            #shouldn't be continuous(actually just i'm too lazy to write, anyway the env is discrete lol)
            #can sample action from continuous actions space and calculate the qsa
            assert 0
            """for _ in range(n_actions):
                pass"""
        v_pi = sum(vals)

        # TODO Calculate Q-Values
        qa_vals = critic.q_net(ob_no)
        q_vals = torch.gather(qa_vals, 1, ac_na.unsqueeze(1)).squeeze(1)
        q_vals = ptu.to_numpy(q_vals)
        assert v_pi.shape == q_vals.shape == (batch_size, )
        advs = q_vals - v_pi
        # TODO Calculate the Advantage
        return advs

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        log = {}

        if self.t > self.num_exploration_steps:
            self.actor.set_critic(self.exploitation_critic)
            self.actor.use_boltzmann = False

        if (self.t > self.learning_starts
                and self.t % self.learning_freq == 0
                and self.replay_buffer.can_sample(self.batch_size)
        ):
            # TODO: Get Reward Weights
            # Get the current explore reward weight and exploit reward weight
            explore_weight = self.explore_weight_schedule.value(self.t)
            exploit_weight = self.exploit_weight_schedule.value(self.t)

            # TODO: Run Exploration Model #
            # Evaluate the exploration model on s' to get the exploration bonus
            # HINT: Normalize the exploration bonus, as RND values vary highly in magnitudeelse:
            expl_bonus = self.exploration_model.forward_np(ob_no)
            expl_bonus = normalize(expl_bonus, np.mean(expl_bonus, axis=0), np.std(expl_bonus, axis=0))  # 其实不用axis=0也行
            assert expl_bonus.shape == re_n.shape, f"{expl_bonus.shape}, {re_n.shape}"

            # TODO: Reward Calculations #
            # Calculate mixed rewards, which will be passed into the exploration critic
            # HINT: See doc for definition of mixed_reward
            mixed_reward = explore_weight * expl_bonus + exploit_weight * re_n

            # TODO: Calculate the environment reward
            # HINT: For part 1, env_reward is just 're_n'
            #       After this, env_reward is 're_n' shifted by self.exploit_rew_shift,
            #       and scaled by self.exploit_rew_scale
            env_reward = (re_n + self.exploit_rew_shift) * self.exploit_rew_scale
            # TODO: Update Critics And Exploration Model #
            # 1): Update the exploration model (based off s')
            # 2): Update the exploration critic (based off mixed_reward)
            # 3): Update the exploitation critic (based off env_reward)
            expl_model_loss = self.exploration_model.update(ob_no)
            exploration_critic_loss = self.exploration_critic.update(ob_no=ob_no, ac_na=ac_na,
                                                                     next_ob_no=next_ob_no, reward_n=mixed_reward,
                                                                     terminal_n=terminal_n)
            exploitation_critic_loss = self.exploitation_critic.update(ob_no=ob_no, ac_na=ac_na,
                                                                       next_ob_no=next_ob_no, reward_n=env_reward,
                                                                       terminal_n=terminal_n)

            # TODO: update actor
            # 1): Estimate the advantage
            # 2): Calculate the awac actor loss
            adv_n = self.estimate_advantage(ob_no, ac_na, re_n, next_ob_no, terminal_n)
            actor_loss = self.awac_actor.update(observations=ob_no, actions=ac_na, adv_n=adv_n)
            # TODO: Update Target Networks #
            if self.num_param_updates % self.target_update_freq == 0:
                #  Update the exploitation and exploration target networks
                self.exploitation_critic.update_target_network()
                self.exploration_critic.update_target_network()
            # Logging #
            log['Exploration Critic Loss'] = exploration_critic_loss['Training Loss']
            log['Exploitation Critic Loss'] = exploitation_critic_loss['Training Loss']
            log['Exploration Model Loss'] = expl_model_loss

            # Uncomment these lines after completing awac
            log['Actor Loss'] = actor_loss

            self.num_param_updates += 1

        self.t += 1
        return log


    def step_env(self):
        """
            Step the env and store the transition
            At the end of this block of code, the simulator should have been
            advanced one step, and the replay buffer should contain one more transition.
            Note that self.last_obs must always point to the new latest observation.
        """
        if (not self.offline_exploitation) or (self.t <= self.num_exploration_steps):
            self.replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs)

        perform_random_action = np.random.random() < self.eps or self.t < self.learning_starts

        if perform_random_action:
            action = self.env.action_space.sample()
        else:
            processed = self.replay_buffer.encode_recent_observation()
            action = self.actor.get_action(processed)

        next_obs, reward, done, info = self.env.step(action)
        self.last_obs = next_obs.copy()

        if (not self.offline_exploitation) or (self.t <= self.num_exploration_steps):
            self.replay_buffer.store_effect(self.replay_buffer_idx, action, reward, done)

        if done:
            self.last_obs = self.env.reset()
