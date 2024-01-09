import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from torch.distributions import Normal
from utils import ReplayBuffer, get_env, run_episode

class NeuralNetwork(nn.Module):
    '''
    This class implements a neural network with a variable number of hidden layers and hidden units.
    You may use this function to parametrize your policy and critic networks.
    '''
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int, hidden_layers: int, activation: str):
        # TODO: Implement this function which should define a neural network 
        # with a variable number of hidden layers and hidden units.
        # Here you should define layers which your network will use.
        super(NeuralNetwork, self).__init__()
        layers = [nn.Linear(input_dim, hidden_size)]
        layers += [nn.Linear(hidden_size, hidden_size) for _ in range(hidden_layers)]
        layers += [nn.Linear(hidden_size, output_dim)]

        self.layers = nn.ModuleList(layers)
        self.activation = activation

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass for the neural network you have defined.
        for layer in self.layers[:-1]:
            if self.activation == 'softmax':
                s = F.softmax(layer(s))
            else:
                s = F.leaky_relu(layer(s))
        return self.layers[-1](s)


class Actor(nn.Module):
    """
    Policy
    """
    def __init__(self, hidden_size: int, hidden_layers: int, actor_lr: float, state_dim: int = 3, action_dim: int = 1, device: torch.device = torch.device('mps')):
        super(Actor, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.actor_lr = actor_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.setup_actor()

    def setup_actor(self):
        '''
        This function sets up the actor network in the Actor class.
        '''
        # TODO: Implement this function which sets up the actor network.
        # Take a look at the NeuralNetwork class in utils.py.
        self.network = NeuralNetwork(
            self.state_dim,
            self.action_dim * 2,
            self.hidden_size,
            self.hidden_layers,
            'relu').to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.actor_lr, amsgrad=True)

    def clamp_log_std(self, log_std: torch.Tensor) -> torch.Tensor:
        '''
        :param log_std: torch.Tensor, log_std of the policy.
        Returns:
        :param log_std: torch.Tensor, log_std of the policy clamped between LOG_STD_MIN and LOG_STD_MAX.
        '''
        return torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
    
    def forward(self, state: torch.Tensor, deterministic: bool) -> (torch.Tensor, torch.Tensor):
        '''
        :param state: torch.Tensor, state of the agent
        :param deterministic: boolean, if true return a deterministic action otherwise sample from the policy distribution.
        Returns:
        :param action: torch.Tensor, action the policy returns for the state.
        :param log_prob: log_probability of the the action.
        '''
        assert state.shape == (3,) or state.shape[1] == self.state_dim, 'State passed to this method has a wrong shape'
        # TODO: Implement this function which returns an action and its log probability.
        # If working with stochastic policies, make sure that its log_std are clamped 
        # using the clamp_log_std function.
        mu_logstd = self.network(state)
        mu, log_prob = mu_logstd.chunk(2, dim=1)
        log_prob = self.clamp_log_std(log_prob)
        std = log_prob.exp()
        
        reparameter = Normal(mu, std)
        x_t = reparameter.rsample()
        y_t = torch.tanh(x_t)
        action = y_t

        # Enforcing Action Bound
        log_prob = reparameter.log_prob(x_t)
        log_prob = log_prob - torch.sum(torch.log((1 - y_t.pow(2)) + 1e-6), dim=-1, keepdim=True)

        assert action.shape == (state.shape[0], self.action_dim) and \
            log_prob.shape == (state.shape[0], self.action_dim), 'Incorrect shape for action or log_prob.'
        # if deterministic:
        #     return torch.tanh(mu), log_prob
        return action, log_prob


class Critic(nn.Module):
    """
    Q-function(s)
    """
    def __init__(self, hidden_size: int,
                hidden_layers: int, critic_lr: int, state_dim: int = 3,
                action_dim: int = 1,device: torch.device = torch.device('cpu')):
        super(Critic, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.critic_lr = critic_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.setup_critic()

    def setup_critic(self):
        self.network = NeuralNetwork(
            self.state_dim + self.action_dim,
            self.action_dim,
            256,
            2,
            'relu')
        self.optimizer = optim.Adam(self.parameters(), lr=self.critic_lr)

    def forward(self, x, a):
        return self.network.forward(torch.cat([x, a], dim=-1))


class TrainableParameter:
    '''
    This class could be used to define a trainable parameter in your method. You could find it 
    useful if you try to implement the entropy temerature parameter for SAC algorithm.
    '''
    def __init__(self, init_param: float, lr_param: float, 
                 train_param: bool, device: torch.device = torch.device('cpu')):
        self.log_param = torch.tensor(np.log(init_param), requires_grad=train_param, device=device)
        self.optimizer = optim.Adam([self.log_param], lr=lr_param)

    def get_param(self) -> torch.Tensor:
        return torch.exp(self.log_param)

    def get_log_param(self) -> torch.Tensor:
        return self.log_param


class Agent:
    def __init__(self):
        # Environment variables. You don't need to change this.
        self.state_dim = 3  # [cos(theta), sin(theta), theta_dot]
        self.action_dim = 1  # [torque] in[-1,1]
        self.batch_size = 200
        self.discount = 0.98
        self.min_buffer_size = 1000
        self.max_buffer_size = 100000
        # If your PC possesses a GPU, you should be able to use it for training, 
        # as self.device should be 'cuda' in that case.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: {}".format(self.device))
        self.memory = ReplayBuffer(self.min_buffer_size, self.max_buffer_size, self.device)
        self.setup_agent()

    def setup_agent(self):
        self.actor = Actor(256, 2, 1e-4, self.state_dim, self.action_dim, self.device)

        self.q1 = Critic(256, 2, 1e-4, self.state_dim, self.action_dim).to(self.device)
        self.q2 = Critic(256, 2, 1e-4, self.state_dim, self.action_dim).to(self.device)
        self.q1_target = Critic(256, 2, 1e-4, self.state_dim, self.action_dim).to(self.device)
        self.q2_target = Critic(256, 2, 1e-4, self.state_dim, self.action_dim).to(self.device)
        
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)

        self.log_alpha = TrainableParameter(0.01, 0.005, True, self.device)

    def critic_target_update(self, base_net: NeuralNetwork, target_net: NeuralNetwork, tau: float, soft_update: bool):
        '''
        This method updates the target network parameters using the source network parameters.
        If soft_update is True, then perform a soft update, otherwise a hard update (copy).
        :param base_net: source network
        :param target_net: target network
        :param tau: soft update parameter
        :param soft_update: boolean to indicate whether to perform a soft update or not
        '''
        for param_target, param in zip(target_net.parameters(), base_net.parameters()):
            if soft_update:
                param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
            else:
                param_target.data.copy_(param.data)

    def choose_action(self, s):
        with torch.no_grad():
            action, log_prob = self.actor(s.to(self.device), False)
        return action, log_prob

    def get_action(self, s: np.ndarray, train: bool) -> np.ndarray:
        """
        :param s: np.ndarray, state of the pendulum. shape (3, )
        :param train: boolean to indicate if you are in eval or train mode. 
                    You can find it useful if you want to sample from deterministic policy.
        :return: np.ndarray,, action to apply on the environment, shape (1,)
        """
        # TODO: Implement a function that returns an action from the policy for the state s.
        # action = np.random.uniform(-1, 1, (1,))
        with torch.no_grad():
            action, _ = self.actor(torch.Tensor(s).view(1,-1).to(self.device), train)
            action = action.cpu().numpy().squeeze(0)
        assert action.shape == (1,), 'Incorrect action shape.'
        assert isinstance(action, np.ndarray ), 'Action dtype must be np.ndarray'
        return np.atleast_1d(action)

    def calc_target(self, mini_batch):
        _, _, r, s_prime = mini_batch
        with torch.no_grad():
            a_prime, log_prob_prime = self.actor(s_prime, False)
            entropy = -self.log_alpha.get_param() * log_prob_prime
            q1_target, q2_target = self.q1_target(s_prime, a_prime), self.q2_target(s_prime, a_prime)
            q_target = torch.min(q1_target, q2_target)
            target = r + self.discount * (q_target + entropy)
        return target
    
    @staticmethod
    def run_gradient_update_step(object, loss: torch.Tensor):
        '''
        This function takes in a object containing trainable parameters and an optimizer, 
        and using a given loss, runs one step of gradient update. If you set up trainable parameters 
        and optimizer inside the object, you could find this function useful while training.
        :param object: object containing trainable parameters and an optimizer
        '''
        object.optimizer.zero_grad()
        loss.mean().backward()
        object.optimizer.step()

    def train_agent(self):
        mini_batch = self.memory.sample(self.batch_size)
        s_batch, a_batch, _, _ = mini_batch

        td_target = self.calc_target(mini_batch)

        q1_loss = F.smooth_l1_loss(self.q1(s_batch, a_batch), td_target)
        self.run_gradient_update_step(self.q1, q1_loss)

        q2_loss = F.smooth_l1_loss(self.q2(s_batch, a_batch), td_target)
        self.run_gradient_update_step(self.q2, q2_loss)

        a, log_prob = self.actor(s_batch, False)
        entropy = -self.log_alpha.get_param() * log_prob
        q1, q2 = self.q1(s_batch, a), self.q2(s_batch, a)
        q = torch.min(q1, q2)
        pi_loss = -(q + entropy)  # for gradient ascent
        self.run_gradient_update_step(self.actor, pi_loss)

        # alpha train
        self.log_alpha.optimizer.zero_grad()
        alpha_loss = -(self.log_alpha.get_param() * (log_prob + -self.action_dim).detach()).mean()
        alpha_loss.backward()
        self.log_alpha.optimizer.step()

        # Q1, Q2 soft-update
        self.critic_target_update(self.q1, self.q1_target, 0.005, True)
        self.critic_target_update(self.q2, self.q2_target, 0.005, True)


if __name__ == '__main__':
    env = get_env(g=10.0, train=True)

    TRAIN_EPISODES = 50
    TEST_EPISODES = 20

    # You may set the save_video param to output the video of one of the evalution episodes, or 
    # you can disable console printing during training and testing by setting verbose to False.
    save_video = True
    verbose = True

    agent = Agent()
    env = get_env(g=10.0, train=True)

    for EP in range(TRAIN_EPISODES):
        run_episode(env, agent, None, verbose, train=True)

    if verbose:
        print('\n')

    test_returns = []
    env = get_env(g=10.0, train=False)

    if save_video:
        video_rec = VideoRecorder(env, "pendulum_episode.mp4")
    
    for EP in range(TEST_EPISODES):
        rec = video_rec if (save_video and EP == TEST_EPISODES - 1) else None
        with torch.no_grad():
            episode_return = run_episode(env, agent, rec, verbose, train=False)
        test_returns.append(episode_return)

    avg_test_return = np.mean(np.array(test_returns))

    print("\n AVG_TEST_RETURN:{:.1f} \n".format(avg_test_return))

    if save_video:
        video_rec.close()