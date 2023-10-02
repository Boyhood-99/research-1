import torch 
import torch.nn as nn
import torch.nn.functional as F

#DDPG network
class Net(nn.Module):
    def __init__(self,state_dim,action_dim) :
        super(Net,self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor = Actor(self.state_dim,self.action_dim)
        self.critic = Critic(self.state_dim,self.action_dim)
  

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim,init_w = 3e-3):
        super(Actor,self).__init__()
      
        self.hidsize = 256
        self.hidsize1 = 128
        # self.hidsize2 = 8
        # 神经网络层layer
        self.layer1 = nn.Linear(state_dim, self.hidsize)
       
        self.layer2 = nn.Linear(self.hidsize,self.hidsize1)
        # self.layer3 = nn.Linear(self.hidsize1,self.hidsize2)
        #输出层
        self.output = nn.Linear(self.hidsize1, action_dim)
        self.output.weight.data.uniform_(-init_w, init_w)
        self.output.bias.data.uniform_(-init_w, init_w)
         # nn.init.normal_(self.layer1.weight, 0., 0.3) # 以正态分布初始化权重，均值0，标准差0.3
        # nn.init.constant_(self.layer1.bias, 0.1)
        # self.output.weight.data.normal_(0.,0.3) # 初始化权重
        # self.output.bias.data.fill_(0.1)

    def forward(self, s):
        #relu激活
        a = F.relu(self.layer1(s))
        a = F.relu(self.layer2(a))
        # a = F.relu(self.layer3(a))
        a = torch.tanh(self.output(a))
        #恢复action
       
        return a

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim,init_w = 3e-3):
        super(Critic,self).__init__()
        self.hidsize  = 256
        self.hidsize1 = 128
        # self.hidsize2 = 8
        # 状态维度层
        self.layer1 = nn.Linear(state_dim+action_dim, self.hidsize)
        self.layer2 = nn.Linear(self.hidsize, self.hidsize1)
        # self.layer3 = nn.Linear(self.hidsize1, self.hidsize2)
        #输出层
        self.output = nn.Linear(self.hidsize1, 1)
        # self.output.weight.data.normal_(0.,0.3) # 初始化权重
        # self.output.bias.data.fill_(0.1)
        self.output.weight.data.uniform_(-init_w, init_w)
        self.output.bias.data.uniform_(-init_w, init_w)

    def forward(self,s,a):
        x = torch.cat([s,a], 1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        # x = F.relu(self.layer3(x))
        q_value = self.output(x)
        return q_value

        
   #SAC     


#SAC actor network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, actions_dim, hidden_size,hidden_size2,hidden_size3, init_w = 3e-3, log_std_min = -20, log_std_max = 2):
        super(PolicyNetwork, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.hidden_size = hidden_size
        self.hidden_size2 = hidden_size2
        # self.hidden_size3 = hidden_size3
        
        
        self.linear1 = nn.Linear(state_dim, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size2)
        # self.linear3 = nn.Linear(self.hidden_size2,self.hidden_size3)
        
        #均值输出层
        self.mean_linear = nn.Linear(self.hidden_size2, actions_dim)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        #log方差输出层
        self.log_std_linear = nn.Linear(self.hidden_size2, actions_dim)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        # x = F.relu(self.linear3(x))
        
        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        #log方差裁剪
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
#SAC double Q-critic network
class CriticTwin(nn.Module):  
    # shared parameter
    '''def __init__(self,  state_dim, action_dim,hidden_size,hidden_size3):
        super().__init__()
        self.net_sa = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU())  # concat(state, action)
        self.net_q1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size3), nn.ReLU(), nn.Linear(hidden_size3, 1))  # q1 value
        self.net_q2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size3), nn.ReLU(), nn.Linear(hidden_size3, 1))  # q2 value'''

    # no shared parameter
    def __init__(self,  state_dim, action_dim,hidden_size,hidden_size2,hidden_size3,init_w = 3e-3):
        super(CriticTwin,self).__init__()
        self.hidden_size = hidden_size
        self.hidden_size2 = hidden_size2
        # self.hidden_size3 = hidden_size3
        self.net_q1 = nn.Sequential(
            nn.Linear(state_dim+action_dim,self.hidden_size),nn.ReLU(),
            nn.Linear(self.hidden_size,self.hidden_size2),nn.ReLU(),
            #  nn.Linear(self.hidden_size2,self.hidden_size3),nn.ReLU(), 
           )
       
        self.net_q2 = nn.Sequential(
            nn.Linear(state_dim+action_dim,self.hidden_size),nn.ReLU(),
            nn.Linear(self.hidden_size,self.hidden_size2),nn.ReLU(),
            # nn.Linear(self.hidden_size2,self.hidden_size3),nn.ReLU(),
          )

        self.out_q1 = nn.Linear(self.hidden_size2,1)
        self.out_q1.weight.data.uniform_(-init_w, init_w)
        self.out_q1.bias.data.uniform_(-init_w, init_w)

        self.out_q2 = nn.Linear(self.hidden_size2,1)
        self.out_q2.weight.data.uniform_(-init_w, init_w)
        self.out_q2.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        
        return torch.add(*self.get_q1_q2(state, action)) / 2.0  # mean Q value

    def get_q_min(self, state, action):
        return torch.min(*self.get_q1_q2(state, action))  # min Q value

    def get_q1_q2(self, state, action):
        x = torch.cat((state,action),dim = 1)
        return self.out_q1(self.net_q1(x)),self.out_q2(self.net_q2(x))


        # tmp = self.net_sa(torch.cat((state, action), dim = 1))
        # return self.net_q1(tmp), self.net_q2(tmp)  # two Q values
   

#SAC Q-network       
class SoftQNetwork(nn.Module):
    def __init__(self, state_dim, actions_dim, hidden_size, init_w = 3e-3):
        super(SoftQNetwork, self).__init__()
        
        self.linear1 = nn.Linear(state_dim + actions_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


#SAC value网络
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w = 3e-3):
        super(ValueNetwork, self).__init__()
         #两层隐藏层
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        
        #输出层初始化
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
    

'''用于分类器时,Sigmoid函数及其组合通常效果更好。
由于梯度消失问题,有时要避免使用sigmoid和tanh函数。
ReLU函数是一个通用的激活函数,目前在大多数情况下使用。
如果神经网络中出现死神经元,那么PReLU函数就是最好的选择。
请记住,ReLU函数只能在隐藏层中使用。'''


'''
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(Actor,self).__init__()
        self.action_bound = torch.tensor(action_bound)

        # 神经网络层layer
        self.layer1 = nn.Linear(state_dim, 30)
        nn.init.normal_(self.layer1.weight, 0., 0.3) # 以正态分布初始化权重，均值0，标准差0.3
        nn.init.constant_(self.layer1.bias, 0.1)
        
        #输出层
        self.output = nn.Linear(30, action_dim)
        self.output.weight.data.normal_(0.,0.3) # 初始化权重
        self.output.bias.data.fill_(0.1)

    def forward(self, s):
        #relu激活
        a = torch.relu(self.layer1(s))
        # 对action进行放缩,实际上a in [-1,1]
        a = torch.tanh(self.output(a))
        #恢复action
        scaled_a = a * self.action_bound
        return scaled_a

'''