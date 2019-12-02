import torch
import numpy as np
import random
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import QNetwork

BUFFER_SIZE = int(1e5)  # replay buffer
BATCH_SIZE = 128        # minibatch
GAMMA = 0.99            # discount factor
TAU = 1e-3              # soft update parameter
LR = 5e-4               # learning rate

UPDATE_EVERY = 4        # C value of DQN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQNAgent:
    """Learning agent"""

    def __init__(self, state_size, action_size, random_seed, hidden_layers=[64, 64]):
        """Initialize an Agent object.

        Params
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random random_seed
            hidden_layers (list of int ; optional): number of each layer nodes
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)


        # Q-Network
        model_params = [state_size, action_size, random_seed, hidden_layers]
        self.qnetwork_local = QNetwork(*model_params).to(device)
        self.qnetwork_target = QNetwork(*model_params).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(),
                                    lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed, device)
        # Initialize time step (for updating every self.update_every steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every self.update_every time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Calculate target value
        self.qnetwork_target.eval()
        with torch.no_grad():
            Q_dash = self.qnetwork_target(next_states)
            Q_dash_max = torch.max(Q_dash, dim=1, keepdim=True)[0]
            y = rewards + gamma * Q_dash_max * (1 - dones)
        self.qnetwork_target.train()

        # Predict Q-value
        self.optimizer.zero_grad()
        Q = self.qnetwork_local(states)
        y_pred = Q.gather(1, actions)

        # TD-error
        loss = torch.sum((y - y_pred)**2)

        # Optimize
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object.

        Params
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            random_seed (int): random random_seed
            device (str): target device name ("cuda:0" or "cpu")
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        experience_fields = ["state", "action", "reward", "next_state", "done"]
        self.experience = namedtuple("Experience", field_names=experience_fields)
        self.seed = random.seed(seed)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory"""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory, k=self.batch_size)
        experiences = [e for e in experiences if e is not None]

        def to_torch(x):
            return torch.from_numpy(np.vstack(x))

        def to_torch_uint8(x):
            return torch.from_numpy(np.vstack(x).astype(np.uint8))

        states = to_torch([e.state for e in experiences]).float()
        actions = to_torch([e.action for e in experiences]).long()
        rewards = to_torch([e.reward for e in experiences]).float()
        next_states = to_torch([e.next_state for e in experiences]).float()
        dones = to_torch_uint8([e.done for e in experiences]).float()

        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the size of internal memory"""
        return len(self.memory)


class Agent(DQNAgent):
    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # Calculate target value
        self.qnetwork_target.eval()
        with torch.no_grad():
            Q_dash_local = self.qnetwork_local(next_states)
            Q_dash_target = self.qnetwork_target(next_states)
            argmax_action = torch.max(Q_dash_local, dim=1, keepdim=True)[1]
            Q_dash_max = Q_dash_target.gather(1, argmax_action)
            y = rewards + gamma * Q_dash_max * (1 - dones)
        self.qnetwork_target.train()

        # Predict Q-value
        self.optimizer.zero_grad()
        Q = self.qnetwork_local(states)
        y_pred = Q.gather(1, actions)

        loss = torch.sum((y - y_pred)**2)
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
