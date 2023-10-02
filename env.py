import math
import numpy as np
from systemmodel import SystemModel
from copy import deepcopy
# from line_profiler import line_profiler

# np.seterr(divide = 'ignore', invalid = 'ignore')

#for DRL ,no
#45,120
#x_range = [-20,4000],y_range = [-200,200]最大活动范围
#distance_penalty = 0.02,time_penalty = 1 
class Environment(object):

    def __init__(self,f_uav_num = 5,epsilon = 0.001,end_reward = 0,time_max = 80,distance_penalty = 0,time_penalty = 0,x_range = [-20,2000],y_range = [-100,100]):#50
        # np.random.seed(0)
        self.f_uav_num  =  f_uav_num             
        self.l_uav_num  =  1
        self.f_uav_H  =  140               # 无人机的飞行高度
        self.l_uav_H  =  150

        self.distance_max = 200
        self.time_max = time_max
        self.end_reward = end_reward
        self.distance_penalty = distance_penalty
        self.time_penalty = time_penalty
        #动作约束
        self.theta_min  =  -math.pi/2
        self.theta_max  = math.pi/2
        self.v_min = 20
        self.v_max = 30#20
        self.x_range = x_range
        self.y_range = y_range


        self.l_v_min = 5
        self.l_v_max = 40

        self.t_uav_f = 2*10**9

        #FL parameters, γ, L, ξ are set as 2, 4, and 0.1 respectively. The learning rate is λ  =  0.25 
        self.gamma = 2
        self.L = 4
        self.ksi = 0.1#  0.1
        self.lamda = 0.25#0.1
        self.epsilon = epsilon           #0.1#0.001
        
        # self.eta = 0.5#0.01
        # self.eta_min = 0.001
        # self.eta_max = 0.78#1

        self.iteration = 5
        self.iteration_min = 1
        self.iteration_max = 30
        
        self.systemmodel = SystemModel(f_uav_num = self.f_uav_num)
        self.l_uav_location = np.zeros(2)
        self.outofdistance = 0

        self.direction_flag_x = np.ones(self.f_uav_num)#方向控制
        self.direction_flag_y = np.ones(self.f_uav_num)#方向控制
        
    # 定义每一个episode的初始状态，实现随机初始化
    
    def reset(self):
        

        # self.distance = np.zeros((self.f_uav_num,1))
        # self.direction = np.zeros((self.f_uav_num,1))
        self.distance = 20*np.random.random(size = (self.f_uav_num,1))
        self.direction = 2*math.pi*np.random.random(size = (self.f_uav_num,1))
        self.distance_direction = np.concatenate((self.distance,self.direction),axis = 1)#使f_uav的距离和方向相邻

        self.local_accuracy_sum = np.zeros(1)
        self.time_total = np.zeros(1)
    
        self.l_uav_location = np.zeros(2)
        self.state = np.hstack((self.distance_direction.reshape(self.f_uav_num*2,),self.local_accuracy_sum,self.time_total))


        return self.state

    def step(self, action):
        
        
        state = self.state
        self.action = action
        l_uav_location = deepcopy(self.l_uav_location)
        time_total = self.time_total
        
        distance_direction = np.array(state[:self.f_uav_num*2]).reshape(self.f_uav_num,2)
        distance = distance_direction[:,0]
        direction = distance_direction[:,1]

        local_accuracy_sum = state[self.f_uav_num*2:self.f_uav_num*2+1]
       
        
        self.action[0] = (self.l_v_max-self.l_v_min)/2*self.action[0]+(self.l_v_max+self.l_v_min)/2
        # self.action[1] = (self.theta_max-self.theta_min)/2*self.action[1]+(self.theta_max+self.theta_min)/2
        self.action[1] = math.pi/2*self.action[1]
        self.action[2] = math.ceil((self.iteration_max-self.iteration_min)/2*self.action[2]+(self.iteration_max+self.iteration_min)/2)
        
        # local_iteration = math.ceil(2/((2-self.L*self.lamda)*self.lamda*self.gamma)*math.log(1/action[2]))
        t_down_ = self.systemmodel.t_down(distance)
        t_up_ = self.systemmodel.ofdma_t_up(distance)
        t_comp = self.systemmodel.t_comp(self.action[2])
        t_agg = self.systemmodel.t_agg(self.t_uav_f)
        #此处无人机飞行时间为当前状态下，底层无人机计算时间(与本次决策相关)和上下行传输时间(只与当前位置有关)
        fly_time = np.max(t_comp+t_down_+t_up_)+t_agg
        next_time_total = time_total+fly_time
        self.time_total = next_time_total

        #环境进行下一步
        #无范围限制，随机方向和速度
        '''
        #底层无人机当前随机的速度和方向
        f_uav_v = np.random.uniform(self.v_min,self.v_max,size = (self.f_uav_num,1))
        f_uav_theta = np.random.uniform(self.theta_min,self.theta_max,size = (self.f_uav_num,1)) 

        # f_uav_theta_norm = np.clip(np.random.normal(size = (self.f_uav_num,1)),-1,1)      
        # f_uav_theta = f_uav_theta_norm*self.theta_max
        
        #底层无人机下一轮绝对坐标
        next_f_uav_location = []
        for i in range(self.f_uav_num):
            for j in range(2):
                if j == 0:
                    next_f_uav_location_x = l_uav_location[0]+distance[i]*math.cos(direction[i])+fly_time*f_uav_v[i]*math.cos(f_uav_theta[i])
                    next_f_uav_location.append(next_f_uav_location_x)
                else:
                    next_f_uav_location_y = l_uav_location[1]+distance[i]*math.sin(direction[i])+fly_time*f_uav_v[i]*math.sin(f_uav_theta[i])
                    next_f_uav_location.append(next_f_uav_location_y)               
        
        next_f_uav_location  =  np.array(next_f_uav_location).reshape(self.f_uav_num, 2)'''

                
        #限制范围
        f_uav_v = np.random.uniform(self.v_min,self.v_max,size = (self.f_uav_num,1))
        # f_uav_theta = np.random.uniform(self.theta_min,self.theta_max,size = (self.f_uav_num,1))
    
        f_uav_theta = []
        #x，y方向反向.
        # # for i in range(self.f_uav_num):
        #     if self.direction_flag_x[i]:
        #         f_uav_theta_i = np.random.uniform(self.theta_min,self.theta_max)
        #         if self.direction_flag_y[i]:
        #             f_uav_theta_i = -f_uav_theta_i if f_uav_theta_i>0 else f_uav_theta_i
        #         else:
        #             f_uav_theta_i = -f_uav_theta_i if f_uav_theta_i<0 else f_uav_theta_i
        #         f_uav_theta.append(f_uav_theta_i)
        #     else:
        #         f_uav_theta_i = np.random.uniform(self.theta_min,self.theta_max)+math.pi
        #         if self.direction_flag_y[i]:
        #             f_uav_theta_i = -f_uav_theta_i if f_uav_theta_i>0 else f_uav_theta_i
        #         else:
        #             f_uav_theta_i = -f_uav_theta_i if f_uav_theta_i<0 else f_uav_theta_i
        #         f_uav_theta.append(f_uav_theta_i)
        
        #x反向，y不变      
        for i in range(self.f_uav_num):
            if self.direction_flag_x[i]:
                f_uav_theta_i = np.random.uniform(self.theta_min,self.theta_max)
                f_uav_theta.append(f_uav_theta_i)
            else:
                f_uav_theta_i = np.random.uniform(self.theta_min,self.theta_max)+math.pi
                f_uav_theta.append(f_uav_theta_i)
             
        f_uav_theta  =  np.array(f_uav_theta)
        

        #底层无人机下一轮绝对坐标
        next_f_uav_location = []
        for i in range(self.f_uav_num):
            for j in range(2):
                if j == 0:
                    next_f_uav_location_x = l_uav_location[0]+distance[i]*math.cos(direction[i])+fly_time*f_uav_v[i]*math.cos(f_uav_theta[i])
                    # if next_f_uav_location_x>self.x_range[1]:
                    #     self.direction_flag_x[i] = 0
                    #     next_f_uav_location_x = self.x_range[1]
                    #     # next_f_uav_location_x = max(min(
                    #     #     l_uav_location[0]+distance[i]*math.cos(direction[i])+fly_time*f_uav_v[i]*math.cos(f_uav_theta[i]),self.x_range[1]),self.x_range[0])
                    # elif next_f_uav_location_x<self.x_range[0]:
                    #     self.direction_flag_x[i] = 1
                    #     next_f_uav_location_x = self.x_range[0]
                    # else:
                    #     pass
                    next_f_uav_location_x = max(min(
                            next_f_uav_location_x,self.x_range[1]),self.x_range[0])
                    next_f_uav_location.append(next_f_uav_location_x)
                else:
                    next_f_uav_location_y = l_uav_location[1]+distance[i]*math.sin(direction[i])+fly_time*f_uav_v[i]*math.sin(f_uav_theta[i])
                    # if next_f_uav_location_y>self.y_range[1]:
                    #     self.direction_flag_y[i] = 1
                    #     next_f_uav_location_y = self.y_range[1]                      
                    # elif next_f_uav_location_y<self.y_range[0]:
                    #     self.direction_flag_y[i] = 0
                    #     next_f_uav_location_y = self.y_range[0]
                    # else:
                    #     pass
        
                    next_f_uav_location_y = max(min(
                        next_f_uav_location_y,self.y_range[1]),self.y_range[0])
                    next_f_uav_location.append(next_f_uav_location_y)

        next_f_uav_location  =  np.array(next_f_uav_location).reshape(self.f_uav_num, 2)
        


        #顶层无人机新坐标

        l_uav_location[0] = l_uav_location[0]+self.action[0]*math.cos(self.action[1])*fly_time
        l_uav_location[1] = l_uav_location[1]+self.action[0]*math.sin(self.action[1])*fly_time
        
        l_uav_location[0] = max(min(
            l_uav_location[0],self.x_range[1]),self.x_range[0])
        l_uav_location[1] = max(min(
            l_uav_location[1],self.y_range[1]),self.y_range[0])
        
        self.l_uav_location = deepcopy(l_uav_location)

        next_distance = []
        next_direction = []
        next_true_distance = []

        #当前相对水平距离和方位角
        for i in range(self.f_uav_num):
            d_ = math.sqrt((next_f_uav_location[i][0]-l_uav_location[0])**2+(next_f_uav_location[i][1]-l_uav_location[1])**2)

            di_ = math.atan2((next_f_uav_location[i][1]-l_uav_location[1]),(next_f_uav_location[i][0]-l_uav_location[0]))

            next_true_distance.append(math.sqrt(d_**2+(self.f_uav_H-self.l_uav_H )**2))
            next_distance.append(d_)
            next_direction.append(di_)

        next_true_distance = np.array(next_true_distance).reshape(self.f_uav_num,1)
        next_distance = np.array(next_distance).reshape(self.f_uav_num,1)
        next_direction = np.array(next_direction).reshape(self.f_uav_num,1)
        
        next_distance_direction = np.concatenate((next_distance,next_direction),axis = 1)
        #当前距离
      
      
        #局部精度公式求和
        yita = math.exp(-self.action[2]*(2-self.L*self.lamda)*self.lamda*self.gamma/2)
        local_accuracy = math.log(1-((1-yita)*self.gamma**2*self.ksi)/(2*self.L**2))
        next_local_accuracy_sum = local_accuracy_sum+local_accuracy
      
        
        #判断下一个观察状态下，本次episode是否结束

        done  =  1 if next_local_accuracy_sum<math.log(self.epsilon) else 0
        done = np.array(done)

        #环境状态改变,下一个state
        next_state = np.hstack((next_distance_direction.reshape(self.f_uav_num*2,),next_local_accuracy_sum,next_time_total))

        self.state = next_state
        #奖励函数求解
    
        # t_up = self.systemmodel.t_up(next_distance)
        # t_down = self.systemmodel.t_down(next_distance)
        
        #日常奖励
        reward1 = 0
        reward2 = np.array((-np.max(t_down_+t_up_+t_comp))*self.systemmodel.p_fly(action[0]))/1000

        # if l_uav_location[0] == self.x_range[1] or l_uav_location[0] == self.x_range[0]:
        #     reward-= self.penalty
        # if l_uav_location[1] == self.y_range[1] or l_uav_location[1] == self.y_range[0]:
        #     reward-= self.penalty

        # if np.sum(next_true_distance>self.distance_max):
        #     reward-= self.distance_penalty
            
        # if np.sum(next_true_distance>self.distance_max):
        #     self.outofdistance+= 1

        #结算奖励
        if done :
            reward2+= self.end_reward
            if self.time_total>self.time_max:
                reward2-= self.time_penalty
            # if self.outofdistance:
            #     reward-= 1
            #     self.outofdistance = 0

        # if np.max(t_down+t_up+t_comp)>0.2:
        #     reward = reward-1
        # reward = np.array(-np.max(t_comp+t_down+t_up)+5) if done else np.array(-np.max(t_comp+t_down+t_up))
        # reward = np.array(-np.max(t_comp+t_down+t_up)+1) if local_accuracy_sum<math.log(2*self.epsilon) else np.array(-np.max(t_comp+t_down+t_up))
        
        # reward = reward.reshape(1,)

        l = deepcopy(self.l_uav_location)
        f = deepcopy(next_f_uav_location)
        d = deepcopy(np.max(next_true_distance))
        reward = reward1+reward2
        
        return self.state, reward, done,np.max(t_up_+t_down_),np.max(t_comp+t_up_+t_down_),l, f,d,


#for stochastic and fixed
'''
class Environment1(object):

    def __init__(self,f_uav_num = 5,epsilon = 0.1,time_max = 30,end_reward = 7):
        self.f_uav_num  =  f_uav_num             
        self.l_uav_num  =  1
        self.f_uav_H  =  140               # 无人机的飞行高度
        self.l_uav_H  =  150

        self.distance_max = 120
        self.time_max = time_max
        self.end_reward = end_reward
        #动作约束
        self.theta_min  =  -math.pi/2
        self.theta_max  = math.pi/2
        self.v_min = 15
        self.v_max = 20#20

        self.l_v_min = 5
        self.l_v_max = 30

        #FL parameters, γ, L, ξ are set as 2, 4, and 0.1 respectively. The learning rate is λ  =  0.25 
        self.gamma = 2
        self.L = 4
        self.ksi = 0.1
        self.lamda = 0.25#0.1
        self.epsilon = epsilon           #0.1#0.001
        
        # self.eta = 0.5#0.01
        # self.eta_min = 0.001
        # self.eta_max = 0.78#1

        self.iteration = 5
        self.iteration_min = 1
        self.iteration_max = 30
        
        self.systemmodel = SystemModel(f_uav_num = self.f_uav_num)
        self.l_uav_location = np.zeros(2)
        
    # 定义每一个episode的初始状态，实现随机初始化
    def reset(self):
        self.time_total = np.zeros(1)
        self.local_accuracy_sum = np.zeros(1)
        self.l_uav_location = np.array([0,0])
        self.f_uav_location = np.zeros((self.f_uav_num,2))
        self.state = np.hstack((self.f_uav_location.reshape(self.f_uav_num*2,),self.local_accuracy_sum,self.time_total))


        return self.state

    def step(self, action):
       
        time_total = self.time_total
        state = self.state
        f_uav_location = state[:self.f_uav_num*2]
        f_uav_location = f_uav_location.reshape((self.f_uav_num,2))
       
        local_accuracy_sum = state[self.f_uav_num*2:self.f_uav_num*2+1]


        action = action
        action[0] = (self.l_v_max-self.l_v_min)/2*action[0]+(self.l_v_max+self.l_v_min)/2
        # self.action[1] = (self.theta_max-self.theta_min)/2*self.action[1]+(self.theta_max+self.theta_min)/2
        action[1] = math.pi/2*action[1]
        
        action[2] = math.ceil((self.iteration_max-self.iteration_min)/2*action[2]+(self.iteration_max+self.iteration_min)/2)
        
        # local_iteration = math.ceil(2/((2-self.L*self.lamda)*self.lamda*self.gamma)*math.log(1/action[2]))


        t_down_ = self.systemmodel.t_down_(f_uav_location,self.l_uav_location)
        t_up_ = self.systemmodel.ofdma_t_up_(f_uav_location,self.l_uav_location)
        t_comp = self.systemmodel.t_comp(action[2])
        
        #此处无人机飞行时间为当前状态下，底层无人机计算时间(与本次决策相关)和上下行传输时间(只与当前位置有关)
        fly_time = np.max(t_comp+t_down_+t_up_)
        next_time_total = time_total+fly_time
        self.time_total = next_time_total

        #根据action，环境进行下一步
       
        #底层无人机当前随机的速度和方向
        f_uav_v_norm = np.random.uniform(size = (self.f_uav_num,1))
        f_uav_theta_norm = np.random.uniform(size = (self.f_uav_num,1))

        f_uav_v = f_uav_v_norm*(self.v_max-self.v_min)+self.v_min
        f_uav_theta = f_uav_theta_norm*(self.theta_max-self.theta_min)+self.theta_min
        
        next_f_uav_location = []
        next_distance = []
        #for location
        for i in range(self.f_uav_num):
            for j in range(2):
                if j == 0:
                    f_uav_location[i,j] = f_uav_location[i,j]+fly_time*f_uav_v[i]*math.cos(f_uav_theta[i])
                else:
                    f_uav_location[i,j] = f_uav_location[i,j]+fly_time*f_uav_v[i]*math.sin(f_uav_theta[i])
        #for distance
        for i in range(self.f_uav_num):
            d_iL =  math.sqrt((f_uav_location[i][0] - self.l_uav_location[0]) ** 2 \
                + (f_uav_location[i][1] - self.l_uav_location[1])**2+(self.f_uav_H-self.l_uav_H )**2)     
            next_distance.append(d_iL)

        next_distance = np.array(next_distance).reshape(self.f_uav_num,)
        next_f_uav_location  =  np.array(f_uav_location).reshape(self.f_uav_num, 2)


        #局部精度公式求和
        yita = math.exp(-action[2]*(2-self.L*self.lamda)*self.lamda*self.gamma/2)
        next_local_accuracy_sum = local_accuracy_sum+math.log(1-((1-yita)*self.gamma**2*self.ksi)/(2*self.L**2))
        
        #判断下一个观察状态下，本次episode是否结束

        done  =  1 if next_local_accuracy_sum<math.log(self.epsilon) else 0
        done = np.array(done)

        #环境状态改变,下一个state
        next_state = np.hstack((next_f_uav_location.reshape(self.f_uav_num*2,),next_local_accuracy_sum,next_time_total))

        self.state = next_state
        #奖励函数求解
    
        # t_up = self.systemmodel.t_up(next_distance)
        # t_down = self.systemmodel.t_down(next_distance)
        
        #日常奖励
        reward = np.array((-np.max(t_down_+t_up_+t_comp))*self.systemmodel.p_fly(action[0]))/1000

        # if np.sum(next_distance>self.distance_max)> = 1:
        #     reward-= 0.1

        #结算奖励
        if done :
            reward+= 7
            if self.time_total>self.time_max:
                reward-= 1

       
        # reward = np.array(-np.max(t_comp+t_down+t_up)+5) if done else np.array(-np.max(t_comp+t_down+t_up))
        # reward = np.array(-np.max(t_comp+t_down+t_up)+1) if local_accuracy_sum<math.log(2*self.epsilon) else np.array(-np.max(t_comp+t_down+t_up))
        
        # reward = reward.reshape(1,)

        l = deepcopy(self.l_uav_location)
        d = deepcopy(np.max(next_distance))
        
        return self.state, reward, done,np.max(t_up_+t_down_),np.max(t_comp+t_up_+t_down_),l, next_f_uav_location,d
'''



#state space based on location
'''
class Environment(object):

    def __init__(self):
        self.f_uav_num  =  5              # 底层无人机数，K
        self.l_uav_num  =  1
        # self.f_uav_H  =  100               # 无人机的飞行高度
        # self.l_uav_H  =  150
        
        #通信参数
        self.rou_0  =  1.42*10**-4        # 1m参考距离的信道增益
        self.alpha = 2                  #自由空间损耗
        self.p_L = 1                     #顶层无人机发射功率
        self.p_i  = 1                   # 底层无人机的发射功率
        self.G_iL = 1                     #天线增益
        self.G_Li = 1    
        self.B  =  10 ** 6               # 总带宽
        self.N_0  =  10**-20.4           # 噪声功率谱密度
        
        #动作约束
        self.theta_min  =  0
        self.theta_max  =  2*math.pi
        self.v_min = 20
        self.v_max = 30

        self.l_v_min = 5
        self.l_v_max = 40

        #联邦学习参数, γ, L, ξ are set as 2, 4, and 0.1 respectively. The learning rate is λ  =  0.25 ?  =  0.10.
        self.gamma = 2
        self.L = 4
        self.ksi = 0.1
        self.lamda = 0.25#0.1
        self.epsilon = 0.1#0.001
        
        self.eta = 0.5#0.01
        self.eta_min = 0.001
        self.eta_max = 0.78#1
        
        self.beta = 0.01
        self.systemmodel = SystemModel()
        
    # 定义每一个episode的初始状态，实现随机初始化
    def reset(self):
        

        #随机初始化底层无人机位置，范围{0，1000}
        #self.f_uav_location = np.random.uniform(0,1000,size = (self.f_uav_num,2))
        self.f_uav_location = np.zeros((self.f_uav_num,2))

        #随机初始底层无人机速度和方向
        self.f_uav_v_norm = np.random.uniform(size = (self.f_uav_num,1))
        self.f_uav_theta_norm = np.random.uniform(size = (self.f_uav_num,1))

        self.f_uav_v = self.f_uav_v_norm*(self.v_max-self.v_min)+self.v_min
        self.f_uav_theta = self.f_uav_theta_norm*(self.theta_max-self.theta_min)+self.theta_min

        # 随机初始化底层无人机 数据样本量大小，范围{1000，2000}
        # self.f_uav_data = np.random.randint(800,1000,size = (self.f_uav_num,1))

        #随机初始化顶层无人机位置
        self.l_uav_location = np.zeros(2)

        #初始化局部迭代精度之和
        self.local_accuracy_sum = np.array(math.log(1.0-((1.0-self.eta)*self.gamma**2*self.ksi)/(2.0*self.L**2)))


        # 创建初始观察值state, 包括底层无人机的位置(f_uav_num, 2)、底层无人机的数据量(f_uav_num, 1)、顶层无人机的位置(f_uav_num, 2)，局部迭代精度，飞行时间

        # self.state = np.hstack((self.f_uav_location.reshape(self.f_uav_num*2,),self.f_uav_v.reshape(self.f_uav_num,),\
        #     self.f_uav_theta.reshape(self.f_uav_num,),self.f_uav_data.reshape(self.f_uav_num,),self.l_uav_location,self.local_accuracy_sum))
        self.state = np.hstack((self.f_uav_location.reshape(self.f_uav_num*2,),self.f_uav_v.reshape(self.f_uav_num,),\
            self.f_uav_theta.reshape(self.f_uav_num,),self.l_uav_location,self.local_accuracy_sum))


        return self.state

    def step(self, action):
        # assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        
        state = self.state
        
        f_uav_location = state[:self.f_uav_num*2]
        f_uav_location = f_uav_location.reshape((self.f_uav_num,2))
        
        f_uav_v = state[self.f_uav_num*2:self.f_uav_num*3]
        f_uav_v = f_uav_v.reshape((self.f_uav_num,1))
        
        f_uav_theta = state[self.f_uav_num*3:self.f_uav_num*4]
        f_uav_theta = f_uav_theta.reshape((self.f_uav_num,1))

        # f_uav_data = state[self.f_uav_num*4:self.f_uav_num*5]
        # f_uav_data = f_uav_data.reshape((self.f_uav_num,1))
        
        l_uav_location_x = state[self.f_uav_num*4:self.f_uav_num*4+1]
        l_uav_location_y = state[self.f_uav_num*4+1:self.f_uav_num*4+2]
        l_uav_location = np.array([l_uav_location_x,l_uav_location_y])

        local_accuracy_sum = state[self.f_uav_num*4+2:self.f_uav_num*4+3]
        
      
        
        action[0] = (self.l_v_max-self.l_v_min)/2*action[0]+(self.l_v_max+self.l_v_min)/2
        action[1] = (self.theta_max-self.theta_min)/2*action[1]+(self.theta_max+self.theta_min)/2
        action[2] = (self.eta_max-self.eta_min)/2*action[2]+(self.eta_max+self.eta_min)/2
        # print(action[2])
        # action[0]* = (self.v_max-self.v_min)+self.v_min
        # action[1]* = (self.theta_max-self.theta_min)+self.theta_min
        # action[2]* = (self.eta_max-self.eta_min)+self.eta_min
        # print(action)
        
        local_iteration = math.ceil(2/((2-self.L*self.lamda)*self.lamda*self.gamma)*math.log(1/action[2]))
        t_down_ = self.systemmodel.t_down(f_uav_location, l_uav_location)
        t_up_ = self.systemmodel.t_up(f_uav_location, l_uav_location)
        t_comp = self.systemmodel.t_comp(local_iteration)
        
        #此处无人机飞行时间为底层无人机计算时间和上下行传输时间
        fly_time = np.max(t_comp+t_down_+t_up_)

        #根据action，环境进行下一步,
        #底层无人机新坐标
        next_f_uav_location = []
        for i in range(self.f_uav_num):
            for j in range(2):
                if j == 0:
                    next_f_uav_location_ = f_uav_location[i][0]+fly_time*f_uav_v[i]*math.cos(f_uav_theta[i])
                else:
                    next_f_uav_location_ = f_uav_location[i][1]+fly_time*f_uav_v[i]*math.sin(f_uav_theta[i])

                next_f_uav_location.append(next_f_uav_location_)
        
        next_f_uav_location  =  np.array(next_f_uav_location).reshape(self.f_uav_num, 2)

        #底层无人机速度和方向在下一个状态依旧是随机的
        next_f_uav_v_norm = np.random.uniform(size = (self.f_uav_num,1))
        next_f_uav_theta_norm = np.random.uniform(size = (self.f_uav_num,1))

        next_f_uav_v = next_f_uav_v_norm*(self.v_max-self.v_min)+self.v_min
        next_f_uav_theta = next_f_uav_theta_norm*(self.theta_max-self.theta_min)+self.theta_min

       
        #数据量不变
        # next_f_uav_data = f_uav_data    #+np.random.uniform(20,50,size = (self.f_uav_num,1))

        #顶层无人机新坐标
        next_l_uav_x = l_uav_location_x+fly_time*action[0]*math.cos(action[1])
        next_l_uav_y = l_uav_location_y+fly_time*action[0]*math.sin(action[1]) 
        next_l_uav_location  =  np.array([next_l_uav_x,next_l_uav_y])

        #局部精度公式求和
        local_accuracy_sum_ = local_accuracy_sum+math.log(1-((1-action[2])*self.gamma**2*self.ksi)/(2*self.L**2))
        next_local_accuracy_sum = np.array([local_accuracy_sum_])
        # print(next_local_accuracy_sum)
        
        #判断该观察状态下，本次episode是否结束

        done  =  1 if next_local_accuracy_sum<math.log(self.epsilon) else 0
        done = np.array(done)
        

        #环境状态改变,下一个state
        next_state = np.hstack((next_f_uav_location.reshape(self.f_uav_num*2,),next_f_uav_v.reshape(self.f_uav_num,),\
            next_f_uav_theta.reshape(self.f_uav_num,),next_l_uav_location.reshape(2,),next_local_accuracy_sum.reshape(1,)))

        self.state = next_state
        #奖励函数求解
    
        t_up = self.systemmodel.t_up(next_f_uav_location, next_l_uav_location)
        t_down = self.systemmodel.t_down(next_f_uav_location, next_l_uav_location)
        
        # reward = np.array((-np.max(t_down+t_up+t_comp))*self.systemmodel.p_fly(action[0]))/1000
        reward = np.array((-np.max(t_down+t_up+t_comp))*self.systemmodel.p_fly(action[0])/1000+10) if done else np.array((-np.max(t_down+t_up+t_comp))*self.systemmodel.p_fly(action[0]))/1000
        if np.max(t_down+t_up+t_comp)>0.2:
            reward = reward-1
        # reward = np.array(-np.max(t_comp+t_down+t_up)+5) if done else np.array(-np.max(t_comp+t_down+t_up))
        # reward = np.array(-np.max(t_comp+t_down+t_up)+1) if local_accuracy_sum<math.log(2*self.epsilon) else np.array(-np.max(t_comp+t_down+t_up))
        
        # reward = reward.reshape(1,)
        return self.state, reward, done,np.max(t_up),np.max(t_comp)

        '''


    
    
    

