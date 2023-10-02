import math
import numpy as np

np.seterr(divide = 'ignore', invalid = 'ignore')

# np.random.seed(1)

#if dataset Dk is fixed or not
# f_uav_num = 5
# a = np.random.randint(800,1000,size = (f_uav_num,1))
class SystemModel(object):
    def __init__(self,f_uav_num = 5):

        self.f_uav_num = f_uav_num              # 底层无人机数，K
        self.f_uav_H = 140               # 无人机的飞行高度
        self.l_uav_H = 150
        
        #通信参数
        self.rou_0 = 1.42*10**-4  # 1m参考距离的信道增益
        # self.rou_0 = 1*10**-5
        self.alpha = 2                    #自由空间损耗
        self.p_L = 1                     #顶层无人机发射功率
        self.p_i  = 1                # 底层无人机的发射功率
        self.G_iL = 1                     #天线增益
        self.G_Li = 1    
        self.B = 10 ** 6               # 广播带宽
        self.N_0 = 10**-20.4           # 噪声功率谱密度
        # self.N_0 = 10**-16
        self.N_B = 10**-9

        self.B_up = 1*10**6  #上行总带宽
        self.B_il = self.B/self.f_uav_num     #FDMA
        # self.subbandwidth = self.B_up/self.M  #OFDMA
        self.subbandwidth = 5*10**4  #OFDMA
        self.M = 20

        self.S_w = 28*1024  #下行传输数据量
        self.S_w_ = 28*1024  #上行传输数据量
        #计算参数
        self.L  = 10000#800                   # 训练1sample需要的CPU计算周期数
        self.f_uav_f = 1 * (10 ** 9)      # 无人机的计算频率
        self.f_uav_data = np.random.randint(800,1000,size = (self.f_uav_num,1))
        

        self.C_T = 100
        
        #self.k = 10 ** -28              # 无人机CPU的电容系数
        
        self.p0 = 80
        self.pi = 89
        self.U_tip = 120
        self.v0 = 4.03
        self.zeta = 0.6
        self.s = 0.05
        self.rou = 1.225
        self.A = 0.5


    #model based on distance
    def ofdma_t_up(self,d,):
        distance = d
        channel_SNR_up = []
        comm_rate_up = []
        t_up = []
        # 计算每个底层无人机与顶层无人机之间的SNR
        for i in range(self.f_uav_num):
            d_il =  math.sqrt(distance[i]**2+(self.f_uav_H-self.l_uav_H )**2)
            SNR_up = self.p_i*self.rou_0*self.G_iL*self.G_iL / (self.N_B*pow(d_il,self.alpha))
            channel_SNR_up.append(SNR_up)

        # channel_SNR_up =  np.array(channel_SNR_up).reshape(self.f_uav_num, 1)

        # 计算每个底层无人机与顶层无人机之间数据的传输速率
        for i in range(self.f_uav_num):
            rate_up =  self.M/self.f_uav_num*self.subbandwidth * math.log2(1 +channel_SNR_up[i])   #标量
            comm_rate_up.append(rate_up)

        # comm_rate_up =  np.array(comm_rate_up).reshape(self.f_uav_num, 1)


        # 计算底层无人机和顶层无人机的通信时延
        for i in range(self.f_uav_num):
            t_up_ = self.S_w_ / comm_rate_up[i]
            t_up.append(t_up_)

        t_up = np.array(t_up).reshape(self.f_uav_num, 1)
        return t_up

    def t_up(self, d):
         # 通信模型,上行信道
        distance = d
        channel_SNR_up = []
        comm_rate_up = []
        t_up = []
        # 计算每个底层无人机与顶层无人机之间的SNR
        for i in range(self.f_uav_num):
            d_il =  math.sqrt(distance[i]**2+(self.f_uav_H-self.l_uav_H )**2)
            SNR_up = self.p_i*self.rou_0*self.G_iL*self.G_iL / (self.B_il*self.N_0*pow(d_il,self.alpha))
            channel_SNR_up.append(SNR_up)

        channel_SNR_up =  np.array(channel_SNR_up,dtype = 'float32').reshape(self.f_uav_num, 1)

        # 计算每个底层无人机与顶层无人机之间数据的传输速率
        for i in range(self.f_uav_num):
            rate_up =  self.B_il * math.log2(1 +channel_SNR_up[i])   #标量
            comm_rate_up.append(rate_up)

        comm_rate_up =  np.array(comm_rate_up,dtype = 'float32').reshape(self.f_uav_num, 1)


        # 计算底层无人机和顶层无人机的通信时延
        for i in range(self.f_uav_num):
            t_up_ = self.S_w_ / comm_rate_up[i]
            t_up.append(t_up_)

        t_up = np.array(t_up).reshape(self.f_uav_num, 1)
        return t_up
    
    
    def t_down(self, d):
         # 通信模型,上行信道
        distance = d
        channel_SNR_down = []
        comm_rate_down = []
        t_down = []
        # 计算每个底层无人机与顶层无人机之间的SNR
        for i in range(self.f_uav_num):
            d_Li =  math.sqrt(distance[i]**2+(self.f_uav_H-self.l_uav_H )**2)
            SNR_up = self.p_L*self.rou_0*self.G_iL*self.G_iL / (self.N_B*pow(d_Li,self.alpha))
            channel_SNR_down.append(SNR_up)

        channel_SNR_down =  np.array(channel_SNR_down,dtype = 'float32').reshape(self.f_uav_num, 1)

        # 传输速率
        for i in range(self.f_uav_num):
            rate_down = self.B * math.log2(1 +channel_SNR_down[i])  
            comm_rate_down.append(rate_down)

        comm_rate_down =  np.array(comm_rate_down,dtype = 'float32').reshape(self.f_uav_num, 1)


        # 通信时延
        for i in range(self.f_uav_num):
            t_down_ = self.S_w / comm_rate_down[i]
            t_down.append(t_down_)

        t_down = np.array(t_down).reshape(self.f_uav_num, 1)

        return t_down

    def t_comp(self,I):

        t_comp = []
        self.I = I
        '''计算模型'''
        #计算时延
        for i in range(self.f_uav_num):
            t_comp_ = self.I*self.L*self.f_uav_data[i] / self.f_uav_f
           
            t_comp.append(t_comp_)
           

        t_comp = np.array(t_comp).reshape(self.f_uav_num, 1)
        

        return t_comp

    def t_agg(self,F):
        t_uav_f = F
        t_agg = self.f_uav_num*self.S_w*self.C_T/t_uav_f
        return t_agg
    def p_fly(self,v):
        P = self.p0*(1+3*v**2/(self.U_tip**2))+self.pi*self.v0/v+0.5*self.zeta*self.s*self.rou*self.A*v**3

        return P
    
    #model based on location
    def ofdma_t_up_(self,f_uav_location,  l_uav_location):

        channel_SNR_up = []
        comm_rate_up = []
        t_up = []
        
        self.f_uav_location = f_uav_location
        self.l_uav_location = l_uav_location
        

        # 通信模型,上行信道
        # 计算每个底层无人机与顶层无人机之间的SNR
        for i in range(self.f_uav_num):
            d_il =  math.sqrt((self.f_uav_location[i][0] - self.l_uav_location[0]) ** 2 \
                + (self.f_uav_location[i][1] - self.l_uav_location[1])**2+(self.f_uav_H-self.l_uav_H )**2)
            SNR_up = self.p_i*self.rou_0*self.G_iL*self.G_iL / (self.N_B*pow(d_il,self.alpha))
            #(self.subbandwidth*self.N_0*pow(d_il,self.alpha))
            channel_SNR_up.append(SNR_up)

        channel_SNR_up =  np.array(channel_SNR_up).reshape(self.f_uav_num, 1)

        # 计算每个底层无人机与顶层无人机之间数据的传输速率
        for i in range(self.f_uav_num):
            rate_up =  self.M/self.f_uav_num*self.subbandwidth * math.log2(1 +channel_SNR_up[i])   #标量
            comm_rate_up.append(rate_up)

        comm_rate_up =  np.array(comm_rate_up).reshape(self.f_uav_num, 1)


        # 计算底层无人机和顶层无人机的通信时延
        for i in range(self.f_uav_num):
            t_up_ = self.S_w_ / comm_rate_up[i]
            t_up.append(t_up_)

        t_up = np.array(t_up).reshape(self.f_uav_num, 1)

        return t_up
    
    def t_up_(self, f_uav_location,  l_uav_location):
        
        channel_SNR_up = []
        comm_rate_up = []
        t_up = []
        
        self.f_uav_location = f_uav_location
        self.l_uav_location = l_uav_location
        

        # 通信模型,上行信道
        # 计算每个底层无人机与顶层无人机之间的SNR
        for i in range(self.f_uav_num):
            d_il =  math.sqrt((self.f_uav_location[i][0] - self.l_uav_location[0]) ** 2 \
                + (self.f_uav_location[i][1] - self.l_uav_location[1])**2+(self.f_uav_H-self.l_uav_H )**2)
            SNR_up = self.p_i*self.rou_0*self.G_iL*self.G_iL / (self.B_il*self.N_0*pow(d_il,self.alpha))
            channel_SNR_up.append(SNR_up)

        channel_SNR_up =  np.array(channel_SNR_up).reshape(self.f_uav_num, 1)

        # 计算每个底层无人机与顶层无人机之间数据的传输速率
        for i in range(self.f_uav_num):
            rate_up =  self.B_il * math.log2(1 +channel_SNR_up[i])   #标量
            comm_rate_up.append(rate_up)

        comm_rate_up =  np.array(comm_rate_up).reshape(self.f_uav_num, 1)


        # 计算底层无人机和顶层无人机的通信时延
        for i in range(self.f_uav_num):
            t_up_ = self.S_w_ / comm_rate_up[i]
            t_up.append(t_up_)

        t_up = np.array(t_up).reshape(self.f_uav_num, 1)

        return t_up

    def t_down_(self, f_uav_location, l_uav_location):

        channel_SNR_down = []
        comm_rate_down = []
        t_down = []

        self.f_uav_location = f_uav_location
        self.l_uav_location = l_uav_location
        # 通信模型,下行信道

        # SNR
        for i in range(self.f_uav_num):
            d_il =  math.sqrt((self.f_uav_location[i][0] - self.l_uav_location[0]) ** 2\
                + (self.f_uav_location[i][1] - self.l_uav_location[1])**2+(self.f_uav_H-self.l_uav_H )**2)
            SNR_down = self.p_L*self.rou_0*self.G_iL*self.G_iL / (self.B*self.N_0*pow(d_il,self.alpha))
            channel_SNR_down.append(SNR_down)

        channel_SNR_down =  np.array(channel_SNR_down).reshape(self.f_uav_num, 1)

        # 传输速率
        for i in range(self.f_uav_num):
            rate_down = self.B * math.log2(1 +channel_SNR_down[i])   #标量
            comm_rate_down.append(rate_down)

        comm_rate_down =  np.array(comm_rate_down).reshape(self.f_uav_num, 1)


        # 通信时延
        for i in range(self.f_uav_num):
            t_down_ = self.S_w / comm_rate_down[i]
            t_down.append(t_down_)

        t_down = np.array(t_down).reshape(self.f_uav_num, 1)

        return t_down



