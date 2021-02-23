import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hist_len):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim + 30, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action
        self.context = Context(hidden_sizes=[30],
                                   input_dim=state_dim + action_dim + 1,
                                   output_dim = 30, # hiddens_conext
                                   only_concat_context = 3,
                                   history_length = hist_len,
                                   action_dim = action_dim,
                                   obsr_dim = state_dim,
                                   device = device
                                   )
        
    def forward(self, state, pre_infos = None, ret_context  = False):
        
        combined = self.context(pre_infos)
        state = torch.cat([state, combined], dim=-1)
        
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = self.max_action * torch.tanh(self.l3(a))
        if ret_context == True:
            return a, combined
        else:
            return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim,hist_len):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim + 30, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim + 30, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

        self.context = Context(hidden_sizes=[30],
                                   input_dim=state_dim + action_dim + 1,
                                   output_dim = 30, # hiddens_conext
                                   only_concat_context = 3,
                                   history_length = hist_len,
                                   action_dim = action_dim,
                                   obsr_dim = state_dim,
                                   device = device
                                   )
        
    def forward(self, state, action, pre_info = None, ret_context = False):
        sa = torch.cat([state, action], 1)

        combined = self.context(pre_info)
        sa = torch.cat([sa, combined], dim=-1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        
        if ret_context == True:
            return q1, q2, combined
        else:
            return q1, q2


    def Q1(self, state, action, pre_info = None, ret_context = False):
        sa = torch.cat([state, action], 1)
        combined = self.context(pre_info)
        sa = torch.cat([sa, combined], dim=-1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        if ret_context == True:
            return q1, combined
        else:
            return q1
        
class Context(nn.Module):
    """
      This layer just does non-linear transformation(s)
    """
    def __init__(self,
                 hidden_sizes = [50],
                 output_dim = None,
                 input_dim = None,
                 only_concat_context = 0,
                 hidden_activation=F.relu,
                 history_length = 1,
                 action_dim = None,
                 obsr_dim = None,
                 device = 'cpu'
                 ):

        super(Context, self).__init__()
        self.only_concat_context = only_concat_context
        self.hid_act = hidden_activation
        self.fcs = [] # list of linear layer
        self.hidden_sizes = hidden_sizes
        self.input_dim = input_dim
        self.output_dim_final = output_dim # count the fact that there is a skip connection
        self.output_dim_last_layer  = output_dim // 2
        self.hist_length = history_length
        self.device = device
        self.action_dim = action_dim
        self.obsr_dim = obsr_dim

        #### build LSTM or multi-layers FF
        if only_concat_context == 3:
            # use LSTM or GRU
            self.recurrent =nn.GRU(self.input_dim,
                               self.hidden_sizes[0],
                               bidirectional = False,
                               batch_first = True,
                               num_layers = 1)

    def init_recurrent(self, bsize = None):
        '''
            init hidden states
            Batch size can't be none
        '''
        # The order is (num_layers, minibatch_size, hidden_dim)
        # LSTM ==> return (torch.zeros(1, bsize, self.hidden_sizes[0]),
        #        torch.zeros(1, bsize, self.hidden_sizes[0]))
        return torch.zeros(1, bsize, self.hidden_sizes[0]).to(self.device)

    def forward(self, data):
        '''
            pre_x : B * D where B is batch size and D is input_dim
            pre_a : B * A where B is batch size and A is input_dim
            previous_reward: B * 1 where B is batch size and 1 is input_dim
        '''
        previous_action, previous_reward, pre_x = data[0], data[1], data[2]
        
        if self.only_concat_context == 3:
            # first prepare data for LSTM
            bsize, dim = previous_action.shape # previous_action is B* (history_len * D)
            pacts = previous_action.view(bsize, -1, self.action_dim) # view(bsize, self.hist_length, -1)
            prews = previous_reward.view(bsize, -1, 1) # reward dim is 1, view(bsize, self.hist_length, 1)
            pxs   = pre_x.view(bsize, -1, self.obsr_dim ) # view(bsize, self.hist_length, -1)
            pre_act_rew = torch.cat([pacts, prews, pxs], dim = -1) # input to LSTM is [action, reward]

            # init lstm/gru
            hidden = self.init_recurrent(bsize=bsize)

            # lstm/gru
            self.recurrent.flatten_parameters()
            _, hidden = self.recurrent(pre_act_rew, hidden) # hidden is (1, B, hidden_size)
            out = hidden.squeeze(0) # (1, B, hidden_size) ==> (B, hidden_size)

            return out

        else:
            raise NotImplementedError

        return None

class TD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        hist_len = -1
    ):
        state_dim = int(state_dim)
        action_dim = int(action_dim)
        hist_len = int(hist_len)
        self.actor = Actor(state_dim, action_dim, max_action, hist_len).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim, hist_len).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.device = device
        self.total_it = 0


    def select_action(self, state, previous_action, previous_reward, previous_state):
        
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        previous_action = torch.FloatTensor(previous_action.reshape(1, -1)).to(self.device)
        previous_reward = torch.FloatTensor(previous_reward.reshape(1, -1)).to(self.device)
        previous_state = torch.FloatTensor(previous_state.reshape(1, -1)).to(self.device)
        pre_info = [previous_action, previous_reward, previous_state]
        
        return self.actor(state, pre_info).cpu().data.numpy().flatten()

    def train_context_TD3(
                self,
                replay_buffer=None,
                batch_size=100):

        '''
            inputs:
                replay_buffer
                iterations episode_timesteps
            outputs:
        '''
        actor_loss_out = 0.0
        critic_loss_out = 0.0

        ### if there is no eough data in replay buffer, then reduce size of iteration to 20:
        #if replay_buffer.size_rb() < iterations or replay_buffer.size_rb() <  self.batch_size * iterations:
        #    temp = int( replay_buffer.size_rb()/ (self.batch_size) % iterations ) + 1
        #    if temp < iterations:
        #        iterations = temp

        for it in range(1):

            ########
            # Sample replay buffer
            ########
            x, y, u, r, d, pu, pr, px, nu, nr, nx = replay_buffer.sample(batch_size)
            obs = torch.FloatTensor(x).to(self.device)
            next_obs = torch.FloatTensor(y).to(self.device)
            action = torch.FloatTensor(u).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)
            mask = torch.FloatTensor(1 - d).to(self.device)
            previous_action = torch.FloatTensor(pu).to(self.device)
            previous_reward = torch.FloatTensor(pr).to(self.device)
            previous_obs = torch.FloatTensor(px).to(self.device)

            # list of hist_actions and hist_rewards which are one time ahead of previous_ones
            # example:
            # previous_action = [t-3, t-2, t-1]
            # hist_actions    = [t-2, t-1, t]
            hist_actions = torch.FloatTensor(nu).to(self.device)
            hist_rewards = torch.FloatTensor(nr).to(self.device)
            hist_obs     = torch.FloatTensor(nx).to(self.device)

            # combine reward and action
            act_rew = [hist_actions, hist_rewards, hist_obs] # torch.cat([action, reward], dim = -1)
            pre_act_rew = [previous_action, previous_reward, previous_obs] #torch.cat([previous_action, previous_reward], dim = -1)

            ########
            # Select action according to policy and add clipped noise
            # mu'(s_t) = mu(s_t | \theta_t) + N (Eq.7 in https://arxiv.org/abs/1509.02971)
            # OR
            # Eq. 15 in TD3 paper:
            # e ~ clip(N(0, \sigma), -c, c)
            ########
            noise = (torch.randn_like(action) * self.policy_noise ).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_obs, act_rew) + noise).clamp(-self.max_action, self.max_action)

            ########
            #  Update critics
            #  1. Compute the target Q value
            #  2. Get current Q estimates
            #  3. Compute critic loss
            #  4. Optimize the critic
            ########

            # 1. y = r + \gamma * min{Q1, Q2} (s_next, next_action)
            # if done , then only use reward otherwise reward + (self.gamma * target_Q)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action, act_rew)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (mask * self.discount * target_Q).detach()

            # 2.  Get current Q estimates
            current_Q1, current_Q2 = self.critic(obs, action, pre_act_rew)

            # 3. Compute critic loss
            # even we picked min Q, we still need to backprob to both Qs
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            critic_loss_out += critic_loss.item()

            # 4. Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            ########
            # Delayed policy updates
            ########
            if it % self.policy_freq == 0:

                # Compute actor loss
                actor_loss = -self.critic.Q1(obs, self.actor(obs, pre_act_rew), pre_act_rew).mean()
                actor_loss_out += actor_loss.item()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()


                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        out = {}
        out['critic_loss'] = critic_loss_out
        out['actor_loss'] = self.policy_freq * actor_loss_out

        # keep a copy of models' params
        #self.copy_model_params()
        return out, None
    
    
    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1

        # Sample replay buffer 
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
