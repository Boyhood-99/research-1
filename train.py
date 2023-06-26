import torch
import numpy as np
from env import Environment,Environment1
from agent import SAC,DDPG
from copy import deepcopy
import matplotlib.pyplot as plt
from normalization import ZFilter
import math

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_EPISODES=400
BATCHSIZE=256

# 策略和算法
# MAX_EPISODES=800
# BATCHSIZE=512#512   #2048，512
# # MEMORY_WARMUP_CAPACITY=10000
# # REPLAYBUFFER=200000


# 策略和算法
# MAX_EPISODES=400
# BATCHSIZE=2048#2048#512   #2048
# MEMORY_WARMUP_CAPACITY=10000#5000
# REPLAYBUFFER=150000#100000



#accuracy and number 
# MAX_EPISODES=200
# BATCHSIZE=128 #256,32
# # MEMORY_WARMUP_CAPACITY=5000#10000#20000
# # REPLAYBUFFER=150000#200000#100000


#学习率
# MAX_EPISODES=500
# BATCHSIZE=256
# MEMORY_WARMUP_CAPACITY=20000
# REPLAYBUFFER=100000

#数量
# MAX_EPISODES=200
# BATCHSIZE=32 #256,32
# MEMORY_WARMUP_CAPACITY=10000#20000
# REPLAYBUFFER=150000#100000

def stoaction():

    env = Environment() 
    episode_reward=[]
    a_dim = 3
    for i in range(MAX_EPISODES):
      
        s = env.reset() 
        ep_reward = 0
        step_sum=0
        #test
        while True:
                #传输时间、一轮时间和无人机位置
              
                t_comm=[]
                t_total=[]
                l_uav_location=[]
                f_uav_location=[]
                d=[]

                step_sum+=1
                
                a = np.random.uniform(low=-1,high=1, size=(a_dim,))
                s_, r, done ,t_comm_,t_total_,l_uav_location_,f_uav_location_,d_= env.step(a)
               

                l_uav_location.append(l_uav_location_)
                f_uav_location.append(f_uav_location_)
                t_comm.append(t_comm_)
                t_total.append(t_total_)
                d.append(d_)
               

                ep_reward += r
                if done :
                    print('Episode:', i, ' Reward: %.4f' % ep_reward, 'Step_sum: %i' % step_sum,' timetotal: %.4f' % env.time_total )
                    
                    break
                

        episode_reward.append(-(ep_reward-7))#减去最终奖励
  
    return episode_reward,t_comm,t_total,l_uav_location,f_uav_location,d



#for trajectory sac

def sac_train_trajectory(policy_lr = 3e-7,path='./SAC/policy_sac_model',I=np.array(-1/29),env=Environment(),capacity=10000,size=300000):
    env = env 
    F_UAV_NUM=env.f_uav_num
    REPLAYBUFFER=size
    MEMORY_WARMUP_CAPACITY=capacity
    END_REWARD=env.end_reward
    # NET_TARGET=torch.load('./2561281281024/ddpg_model2')
    
    s_dim = F_UAV_NUM*2+2
    a_dim = 2
    
    z=ZFilter(s_dim)
    #a_bound = env.action_space.high
    sac = SAC(state_dim=s_dim,action_dim=a_dim,device=DEVICE,batch_size=BATCHSIZE,
    replay_buffer_size=REPLAYBUFFER,policy_lr = policy_lr)
    episode_reward=[]
   
    t_comm=[]
    t_total=[]
    l_uav_location=[]
    f_uav_location=[]
    d=[]

    entropy=[]
    q=[]
    p=[]
    alpha=[]
    for i in range(MAX_EPISODES):
        a_=[]
        s = env.reset() 
        s=z(s)
        ep_reward = 0
        step_sum=0
        #test
        if i == MAX_EPISODES-1:
            while True:
                
                step_sum+=1
                a = sac.test_choose_action(s) 
                # I=np.random.randint(1,30)
                I=I
                action=np.append(a,I)
                
                s_, r, done ,t_comm_,t_total_,l_uav_location_,f_uav_location_,d_= env.step(action)
                s_=z(s_)

                l_uav_location.append(l_uav_location_)
                f_uav_location.append(f_uav_location_)
                t_comm.append(t_comm_)
                t_total.append(t_total_)
                d.append(d_)
                a_.append(action)

                ep_reward += r
                if done :
                    print('Episode:', i, ' Reward: %.4f' % ep_reward, 'Step_sum: %i' % step_sum )
                    torch.save(sac.policy_net,path)
                    print(a_)
                    break
                s=s_
    #train   
        else:  
            while True:
                step_sum+=1
                a = sac.choose_action(s)
                # I=np.random.randint(1,20)
                I=I
                action=np.append(a,I)
                s_, r, done ,_,_,_,_,_= env.step(action)           
                s_=z(s_)
                sac.replay_buffer.store_transition(s,a,r,s_,done)
                ep_reward += r
    
                if done :
                    if sac.replay_buffer.__len__() > MEMORY_WARMUP_CAPACITY : 
                        for j in range(200) :
                            q_,p_,alpha_=sac.learn()
                            # entropy.append(log)
                            q.append(q_)
                            p.append(p_)
                            alpha.append(alpha_)
                    print('Episode:', i, ' Reward: %.4f' % ep_reward, 'Step_sum: %i' % step_sum,' timetotal: %.4f' % env.time_total ) 
                    break  
                s = s_    

        episode_reward.append(-(ep_reward-END_REWARD))#减去最终奖励
  
    return episode_reward,q,p,alpha,t_comm,t_total,l_uav_location,f_uav_location,d

def sac_train_federated(policy_lr = 3e-7,path='./SAC/policy_sac_model',env=Environment(),capacity=10000,size=300000):
    env = env 
    F_UAV_NUM=env.f_uav_num
    REPLAYBUFFER=size
    MEMORY_WARMUP_CAPACITY=capacity
    END_REWARD=env.end_reward

    s_dim = F_UAV_NUM*2+2
    a_dim = 1
    z=ZFilter(s_dim)
    #a_bound = env.action_space.high
    sac = SAC(state_dim=s_dim,action_dim=a_dim,device=DEVICE,batch_size=BATCHSIZE,
    replay_buffer_size=REPLAYBUFFER,policy_lr = policy_lr)
    episode_reward=[]
   
    t_comm=[]
    t_total=[]
    l_uav_location=[]
    f_uav_location=[]
    d=[]

    entropy=[]
    q=[]
    p=[]
    alpha=[]
    for i in range(MAX_EPISODES):
        a_=[]
        s = env.reset() 
        s=z(s)
        ep_reward = 0
        step_sum=0
        #test
        if i == MAX_EPISODES-1:
            while True:
                step_sum+=1
                a = sac.test_choose_action(s) 
                #随机或者固定
                # v_theta=np.random.uniform(size=(2))
                v_theta=[-2/7,0]
                action=np.concatenate([v_theta,a])
               
                s_, r, done ,t_comm_,t_total_,l_uav_location_,f_uav_location_,d_= env.step(action)

                s_=z(s_)

                l_uav_location.append(l_uav_location_)
                f_uav_location.append(f_uav_location_)
                t_comm.append(t_comm_)
                t_total.append(t_total_)
                # d.append(d_)
                a_.append(action)

                ep_reward += r
                if done :
                    print('Episode:', i, ' Reward: %.4f' % ep_reward, 'Step_sum: %i' % step_sum )
                    torch.save(sac.policy_net,path)
                    print(a_)
                    break
                s=s_
    #train   
        else:  
            while True:
                step_sum+=1
                a = sac.choose_action(s)
                # v_theta=np.random.uniform(size=(2))
                v_theta=[-2/7,0]
                action=np.concatenate([v_theta,a])
                
                s_, r, done ,_,_,_,_,_= env.step(action)
               
                s_=z(s_)
                sac.replay_buffer.store_transition(s,a,r,s_,done)
                ep_reward += r
    
                if done :
                    if sac.replay_buffer.__len__() > MEMORY_WARMUP_CAPACITY : 
                        for j in range(200) :
                            q_,p_,alpha_=sac.learn()
                            # entropy.append(log)
                            q.append(q_)
                            p.append(p_)
                            alpha.append(alpha_)
                    print('Episode:', i, ' Reward: %.4f' % ep_reward, 'Step_sum: %i' % step_sum,' timetotal: %.4f' % env.time_total ) 
                    break  
                s = s_    

        episode_reward.append(-(ep_reward-END_REWARD))#减去最终奖励
  
    return episode_reward,q,p,alpha,t_comm,t_total,l_uav_location,f_uav_location,d

#for sac trajectory and federated optimization
def sac_train(policy_lr = 3e-7,path='./SAC/policy_sac_model',env=Environment(),capacity=10000,size=300000):#3e-7
    env = env
    REPLAYBUFFER=size
    MEMORY_WARMUP_CAPACITY=capacity
    F_UAV_NUM=env.f_uav_num
    END_REWARD=env.end_reward
    s_dim = F_UAV_NUM*2+2
    a_dim = 3
    z=ZFilter(s_dim)
    #a_bound = env.action_space.high
    sac = SAC(state_dim=s_dim,action_dim=a_dim,device=DEVICE,batch_size=BATCHSIZE,
    replay_buffer_size=REPLAYBUFFER,policy_lr = policy_lr)
    episode_reward=[]
   
    t_comm=[]
    t_total=[]
    l_uav_location=[]
    f_uav_location=[]
    # d=[]
    max_distance=[]
    total_time=[]

    entropy=[]
    q=[]
    p=[]
    alpha=[]
    for i in range(MAX_EPISODES):
        a_=[]
        s = env.reset() 
        s=z(s)
        ep_reward = 0
        step_sum=0
        #test
        if i == MAX_EPISODES-1:
            while True:
                step_sum+=1
                a = sac.test_choose_action(s) 
                # a = sac.choose_action(s)             
                # s_, r, done ,x,y= env.step(a)
                             
                s_, r, done ,t_comm_,t_total_,l_uav_location_,f_uav_location_,_= env.step(a)
                action=deepcopy(a)
                s_=z(s_)

                l_uav_location.append(l_uav_location_)
                f_uav_location.append(f_uav_location_)
                t_comm.append(t_comm_)
                t_total.append(t_total_)
                # d.append(d_)
                a_.append(action)

                ep_reward += r
                if done :
                    print('Episode:', i, ' Reward: %.4f' % ep_reward, 'Step_sum: %i' % step_sum )
                    torch.save(sac.policy_net,path)
                    print(a_)
                    break
                s=s_
    #train   
        else:  
            while True:
                d=[]
                
                step_sum+=1
                a = sac.choose_action(s)
                action=deepcopy(a)
                s_, r, done ,_,_,_,_,d_= env.step(action)
                # s_, r, done ,x,y= env.step(a)
                s_=z(s_)
                sac.replay_buffer.store_transition(s,a,r,s_,done)

                ep_reward += r
                d.append(d_)



                # #回合更新
                if done :
                    if sac.replay_buffer.__len__() > MEMORY_WARMUP_CAPACITY : 
                        for j in range(200):   #50-200np.clip((200-i)*2,50,200)
                            q_,p_,alpha_=sac.learn()
                            # entropy.append(log)
                            q.append(q_)
                            p.append(p_)
                            alpha.append(alpha_)
                    print('Episode:', i, ' Reward: %.4f' % ep_reward, 'Step_sum: %i' % step_sum,' timetotal: %.4f' % env.time_total ) 
                    max_distance.append(np.max(d))
                    total_time.append(env.time_total)
                    # print(env.flytime)
                    break  


                #单步更新
                
                # if sac.replay_buffer.__len__() > MEMORY_WARMUP_CAPACITY : 
                #     q_,p_,alpha_=sac.learn()
                #     # entropy.append(log)
                #     q.append(q_)
                #     p.append(p_)
                #     alpha.append(alpha_)
                # if done:
                #     print('Episode:', i, ' Reward: %.4f' % ep_reward, 'Step_sum: %i' % step_sum,' timetotal: %.4f' % env.time_total )  
                #     max_distance.append(np.max(d))
                #     break  

                s = s_    
        
        if env.time_total>env.time_max:
            ep_reward+=env.time_penalty
        episode_reward.append(-(ep_reward-END_REWARD))#减去最终奖励
        


        # episode_reward.append(-ep_reward)
  
    return episode_reward,q,p,alpha,t_comm,t_total,l_uav_location,f_uav_location,max_distance,total_time


def ddpg_train(lr_a=3e-7, lr_c=1e-5,path='./ddpg_model',env=Environment(),capacity=10000,size=300000):
    NOISE=0.8
    REPLACEMENT = [dict(name='soft', tau=0.001),dict(name='hard', rep_iter=600)][0] 
    env = env
    REPLAYBUFFER=size
    MEMORY_WARMUP_CAPACITY=capacity
    F_UAV_NUM=env.f_uav_num
    END_REWARD=env.end_reward
    s_dim = F_UAV_NUM*2+2
    a_dim = 3
    z=ZFilter(s_dim)
    ddpg = DDPG(state_dim=s_dim,action_dim=a_dim,device=DEVICE,replacement=REPLACEMENT,
                replay_buffer_size=REPLAYBUFFER,batch_size=BATCHSIZE,lr_a=lr_a, lr_c=lr_c
                )

    t_comm=[]
    t_total=[]
    l_uav_location=[]
    f_uav_location=[]
    d=[]

    episode_reward=[]
    a_loss=[]
    td_error=[]
   
    for i in range(MAX_EPISODES):
        
        a_=[]
        s = env.reset() 
        s=z(s)
        ep_reward = 0
        step_sum=0
        if i == MAX_EPISODES-1:  #test
            while True:
                step_sum+=1
                a = ddpg.test_choose_action(s)     
                s_, r, done ,t_comm_,t_total_,l,f,d_= env.step(a)
                action=deepcopy(a)

                l_uav_location.append(l)
                f_uav_location.append(f)
                t_comm.append(t_comm_)
                t_total.append(t_total_)
                d.append(d_)
                a_.append(action)

                s_=z(s_)
                ep_reward += r
                if done :
                    print('Episode:', i, ' Reward: %.4f' % ep_reward, 'Explore: %.2f' % NOISE,'Step_sum: %i' % step_sum )
                    torch.save(ddpg.net_target,path)
                    print(a_)
                    break
                s = s_
    #train
        else:
            while True:
                step_sum+=1
                a = ddpg.choose_action(s)
                a = np.clip(np.random.normal(a, NOISE), -1,1) 
                action=deepcopy(a) 
                s_, r, done ,_,_,_,_,d_= env.step(action)

                s_=z(s_)
                ep_reward += r
                ddpg.replay_buffer.store_transition(s,a,r,s_,done)
                                       
                if ddpg.replay_buffer.__len__() > MEMORY_WARMUP_CAPACITY and done:
                    NOISE*=0.99

                #回合更新
                if  done  :
                    if ddpg.replay_buffer.__len__() > MEMORY_WARMUP_CAPACITY :
                        for j in range(200):                
                            a_loss_,td_error_=ddpg.learn()
                            a_loss.append(math.fabs(a_loss_.cpu().detach().numpy()))
                            td_error.append(td_error_.cpu().detach().numpy())  
                       
                    print('Episode:', i, ' Reward: %.4f' % ep_reward, 'Explore: %.2f' % NOISE,
                    'Step_sum: %i' % step_sum ,' timetotal: %.4f' % env.time_total)  
                    break      
                
                #单步更新
                # if ddpg.replay_buffer.__len__() > MEMORY_WARMUP_CAPACITY :                              
                #     a_loss_,td_error_=ddpg.learn()
                #     a_loss.append(math.fabs(a_loss_.cpu().detach().numpy()))
                #     td_error.append(td_error_.cpu().detach().numpy())  
                # if done:
                #     NOISE*=0.99
                #     print('Episode:', i, ' Reward: %.4f' % ep_reward, 'Explore: %.2f' % NOISE,
                #     'Step_sum: %i' % step_sum ,' timetotal: %.4f' % env.time_total)  
                #     break   

                s = s_

        episode_reward.append(-(ep_reward-END_REWARD))#减去最终奖励
  
    return episode_reward,a_loss,td_error,t_comm,t_total,l_uav_location,f_uav_location,d




def fixaction():
    env = Environment() 
    episode_reward=[]
    for i in range(MAX_EPISODES):
        s = env.reset() 
        ep_reward = 0
        step_sum=0
        l_uav_location=[]
        f_uav_location=[]
        while True:
                #传输时间、一轮时间和无人机位置
              
                t_comm=[]
                t_total=[]
                
                d=[]

                step_sum+=1
                
                # a = np.array([-0.7,0,1]) #I=30，
                # a = np.array([-0.7,0,-1]) #I=1，
                # a = np.array([-0.7,0,0.30]) #I=20，
                # a = np.array([-0.7,0,-0.04])  #I=15，
                # a = np.array([-0.7,0,-0.38]) #I=10，
                # a = np.array([-0.6,0,-0.73])  #I=5
                # a = np.array([-0.3,0,-0.93])  #I=2，
                a = np.array([-0.4,0,-0.38])  #I=2，

                s_, r, done ,t_comm_,t_total_,l_uav_location_,f_uav_location_,d_= env.step(a)
               

                l_uav_location.append(l_uav_location_)
                f_uav_location.append(f_uav_location_)
                t_comm.append(t_comm_)
                t_total.append(t_total_)
                d.append(d_)

                ep_reward += r
                if done :
                    print('Episode:', i, ' Reward: %.4f' % ep_reward, 'Step_sum: %i' % step_sum,' timetotal: %.4f' % env.time_total )
                    break
                

        episode_reward.append(-ep_reward)#减去最终奖励
  
    return episode_reward,t_comm,t_total,l_uav_location,f_uav_location,d


def draw():
    f_uav_num=5
    
    Q_LOSS_FLAG=0
    Q_VALUE_FLAG=0

    DDPG_TRA_FLAG=0
    SAC_TRA_FLAG=1
    FIX_TRA_FLAG=0

    ALGRITHM_FLAG=1
    POLICY_FLAG=0
    LEARNING_RATE=0

    TIME_FLAG=1
    DISTANCE_FLAG=0

    TRA_COMPARISION=0

    F_UAV_NUM_COM=0
    ACCURACY_FLAG=0

    BAR_FLAG=0

    
    
    # episode_reward_fix,t_comm,t_total,l_uav_location_f,f_uav_location_f,d_fix=fixaction()
    # episode_reward_sto,t_comm,t_total,l_uav_location,f_uav_location,d_sto=stoaction()
    # episode_reward_sac_fe,q_fe,_,_,_,_,_,_,_=sac_train_federated()
    
    # episode_reward_sac_tra15,_,_,_,_,_,_,_,_=sac_train_trajectory()#15
    # episode_reward_sac_tra10,_,_,_,_,_,_,_,_=sac_train_trajectory(I=-11/29)#10


    episode_reward_sac,q,p,alpha,t_comm_sac,t_total_sac,l_uav_location_sac,f_uav_location_sac,d_sac,total_time_sac=sac_train()
    # episode_reward_ddpg,a_loss,td_error,t_comm_ddpg,t_total_ddpg,l_uav_location_ddpg,f_uav_location_ddpg,d_ddpg=ddpg_train()

   

    #学习率
    # episode_reward_sac_1e_4,_,_,_,_,_,_,_,_=sac_train(policy_lr = 1e-4,path='./SAC/policy_sac_model')
    # episode_reward_sac_3e_6,_,_,_,_,_,_,_,_=sac_train(policy_lr = 3e-6,path='./SAC/policy_sac_model')
    # episode_reward_sac_1e_5,_,_,_,_,_,_,_,_=sac_train(policy_lr = 1e-5,path='./SAC/policy_sac_model')


    #无人机数量
    # episode_reward_sac_10,q,p,alpha,t_comm_sac,t_total_sac,l_uav_location_sac,f_uav_location_sac,d_sac=sac_train(env=Environment(f_uav_num=10),f_uav_num=10)
    # episode_reward_sac_15,q,p,alpha,t_comm_sac,t_total_sac,l_uav_location_sac,f_uav_location_sac,d_sac=sac_train(env=Environment(f_uav_num=15),f_uav_num=15)
    # episode_reward_sac_20,q,p,alpha,t_comm_sac,t_total_sac,l_uav_location_sac,f_uav_location_sac,d_sac=sac_train(env=Environment(f_uav_num=20),f_uav_num=20)


    #模型精度
    # episode_reward_sac_001,_,_,_,_,_,_,_,_=sac_train(env=Environment(epsilon=0.01,time_max=50,end_reward=15))
    # episode_reward_sac_0005,_,_,_,_,_,_,_,_=sac_train(env=Environment(epsilon=0.005,time_max=60,end_reward=17.5))
    # episode_reward_sac_0001,_,_,_,_,_,_,_,_=sac_train(env=Environment(epsilon=0.001,time_max=60,end_reward=22))

# accuracy,number
    # episode_reward_sac_5_01,_,_,_,_,_,_,_,_,_=sac_train(env=Environment(epsilon=0.1))
    # episode_reward_sac_5_001,_,_,_,_,_,_,_,_,_=sac_train(env=Environment(epsilon=0.01))
    # episode_reward_sac_5_0005,_,_,_,_,_,_,_,_,_=sac_train(env=Environment(epsilon=0.005))#基准设置

    # episode_reward_sac_10_01,_,_,_,_,_,_,_,_,_=sac_train(env=Environment(f_uav_num=10,epsilon=0.1))
    # episode_reward_sac_10_001,_,_,_,_,_,_,_,_,_=sac_train(env=Environment(f_uav_num=10,epsilon=0.01))
    # episode_reward_sac_10_0005,_,_,_,_,_,_,_,_,_=sac_train(env=Environment(f_uav_num=10,epsilon=0.005,x_range=[-20,3000],y_range=[-200,200]))
    # episode_reward_sac_10_0001,_,_,_,_,_,_,_,_,_=sac_train(env=Environment(f_uav_num=10,x_range=[-20,3000],y_range=[-200,200]))

    # episode_reward_sac_20_01,_,_,_,_,_,_,_,_,_=sac_train(env=Environment(f_uav_num=20,epsilon=0.1,x_range=[-20,1000],y_range=[-200,200]))
    # episode_reward_sac_20_001,_,_,_,_,_,_,_,_,_=sac_train(env=Environment(f_uav_num=20,epsilon=0.01,x_range=[-20,2000],y_range=[-200,200]),capacity=10000)
    # episode_reward_sac_20_0005,_,_,_,_,_,_,_,_,_=sac_train(env=Environment(f_uav_num=20,epsilon=0.005,x_range=[-20,3000],y_range=[-200,200]),capacity=12000)
    # episode_reward_sac_20_0001,_,_,_,_,_,_,_,_,_=sac_train(env=Environment(f_uav_num=20,x_range=[-20,4000],y_range=[-200,200]),capacity=15000)

    if FIX_TRA_FLAG:
        l_uav_location_f=np.array(l_uav_location_f)     #.reshape(-1,2) #第0维
        

        # f_uav_location_sac=f_uav_location_sac[initial_location:initial_location+trajectory_length]
        f_uav_location_f=np.array(f_uav_location_f)  #.reshape(-1,f_uav_num,2) 

    #SAC trajectory

        plt.figure(20)
        ax = plt.axes(projection='3d')

        l_uav_location_f_x=l_uav_location_f[:,0]
        l_uav_location_f_y=l_uav_location_f[:,1]

        ax.plot3D(l_uav_location_f_x, l_uav_location_f_y, 150, label='The trajectory of T-UAV')
        ax.legend()

        for i in range(f_uav_num):
            f_uav_location_f_x=f_uav_location_f[:,i,0]
            f_uav_location_f_y=f_uav_location_f[:,i,1]
            ax.plot3D(f_uav_location_f_x, f_uav_location_f_y, 140 )
        ax.set_title('3D line plot')
        plt.savefig('./SAC/path_f.jpg')
        plt.savefig('./SAC/path_f.eps')
        plt.savefig('./SAC/path_f.pdf')
    
    if BAR_FLAG:
        initial_point=350
        #柱状图
        epsilon01=[np.array(episode_reward_sac_5_01[initial_point:]).mean(),np.array(episode_reward_sac_10_01[initial_point:]).mean(),]
                   #np.array(episode_reward_sac_20_01[initial_point:]).mean()]
        epsilon001=[np.array(episode_reward_sac_5_001[initial_point:]).mean(),np.array(episode_reward_sac_10_001[initial_point:]).mean(),]
                    # np.array(episode_reward_sac_20_001[initial_point:]).mean()]
        epsilon0005=[np.array(episode_reward_sac_5_0005[initial_point:]).mean(),np.array(episode_reward_sac_10_0005[initial_point:]).mean(),]
                    #  np.array(episode_reward_sac_20_0005[initial_point:]).mean()]
        epsilon0001=[np.array(episode_reward_sac[initial_point:]).mean(),np.array(episode_reward_sac_10_0001[initial_point:]).mean(),]
                    #  np.array(episode_reward_sac_20_0001[initial_point:]).mean()]

        # num_f_uav_5=[epsilon01[0],epsilon001[0],epsilon0005[0],epsilon0001[0]]
        # num_f_uav_10=[epsilon01[1],epsilon001[1],epsilon0005[1],epsilon0001[1]]
        
        num_f_uav=['K=5','K=10']
        xticks=np.arange(len(num_f_uav))

        fig,ax=plt.subplots(dpi=200)#画布大小，分辨率；
        
        width=0.2
        ax.bar(xticks,epsilon01,width=width,label='ε=0.1')
        ax.bar(xticks+0.2,epsilon001,width=width,label='ε=0.01')
        ax.bar(xticks+0.4,epsilon0005,width=width,label='ε=0.005')
        ax.bar(xticks+0.6,epsilon0001,width=width,label='ε=0.001')

        ax.set_xlabel("Number of L-UAVs")
        ax.set_ylabel("Total energy(kJ)")
        
        # plt.rcParams.update({'font.size': 15})
        ax.legend()
        ax.set_xticks(xticks+0.3)
        ax.set_xticklabels(num_f_uav)

        # x = [5,10,20]
        # x1=np.array([i for i in range(0,15,5)])
        # #将每四个柱状图之间空一格
        # x2=x1+1
        # x3=x1+2
        # x4=x1+3
        # x5=x1+4
        
        # y2 = epsilon01
        # y3 = epsilon001
        # y4 = epsilon0005
        # y5 = epsilon0001
        
        # plt.bar(x1,y2,width=width,label='SAC(ε=0.1)')
        
        # plt.bar(x2,y3,width=width,label='SAC(ε=0.01)')
        
        # plt.bar(x3,y4,width=width,label='SAC(ε=0.005)')
        # plt.bar(x4,y5,width=width,label='SAC(ε=0.001)')
        # plt.bar(x5,0,width=width) #空格一个
        
        # plt.xlabel('Number of B-UAVs')
        # plt.ylabel('Total Energy(kJ)')
        # plt.legend()
        # plt.xticks(x1+1.5,x,rotation = 45)#+1.5是让下标在四个柱子中间
      
        
        '''#每一个柱上添加相应值
        for a,b,c,d,e,f,g,h in zip(x1,x2,x3,x4,y2,y3,y4,y5):
            plt.text(a,e+100,int(e),fontsize=4,ha='center')
            plt.text(b,f+100,int(f),fontsize=4,ha='center')
            plt.text(c,g+100,int(g),fontsize=4,ha='center')
            plt.text(d,h+100,int(h),fontsize=4,ha='center')'''
        
        

        plt.savefig('./SAC/bar.eps')
        plt.savefig('./SAC/bar.pdf')
        plt.savefig('./SAC/bar.jpg')


    if ACCURACY_FLAG:
        plt.figure(14,dpi=200)

        plt.plot(episode_reward_sac,color='r', linewidth=1, linestyle='-',label='SAC(ε=0.1)')
        plt.plot(episode_reward_sac_001,color='y', linewidth=1, linestyle='-',label='SAC(ε=0.01)')
        # plt.plot(episode_reward_sac_15,color='b', linewidth=1, linestyle='-',label='SAC(B-UAVs=15)')
        plt.plot(episode_reward_sac_0001,color='g', linewidth=1, linestyle='-',label='SAC(ε=0.001)')

        plt.xlabel('Episodes')
        plt.ylabel('Total energy(kJ)')
        plt.legend()

        plt.savefig('./SAC/accuracy.jpg')


    if F_UAV_NUM_COM:
        plt.figure(13,dpi=200)

        plt.plot(episode_reward_sac,color='r', linewidth=1, linestyle='-',label='SAC(B-UAVs=5)')
        plt.plot(episode_reward_sac_10,color='y', linewidth=1, linestyle='-',label='SAC(B-UAVs=10)')
        # plt.plot(episode_reward_sac_15,color='b', linewidth=1, linestyle='-',label='SAC(B-UAVs=15)')
        plt.plot(episode_reward_sac_20,color='g', linewidth=1, linestyle='-',label='SAC(B-UAVs=20)')

        plt.xlabel('Episodes')
        plt.ylabel('Total Energy(kJ)')
        plt.legend()


        plt.savefig('./SAC/num.jpg')

    
    if LEARNING_RATE:
        plt.figure(12,dpi=200)
        plt.plot(episode_reward_sac,color='b', linewidth=1, linestyle='-',label='SAC,lr_a=3e-5')
        # plt.plot(episode_reward_sac_1e_4,color='r', linewidth=1, linestyle='-',label='SAC,lr_a=1e-4')
        plt.plot(episode_reward_sac_1e_5,color='g', linewidth=1, linestyle='-',label='SAC,lr_a=1e-5')
        plt.plot(episode_reward_sac_3e_6,color='y', linewidth=1, linestyle='-',label='SAC,lr_a=3e-6')

        plt.xlabel('Episodes')
        plt.ylabel('Total Energy(kJ)')
        plt.legend()

        plt.savefig('./SAC/lr_a.jpg')

    if ALGRITHM_FLAG:
        plt.figure(1,dpi=200)

        # plt.plot(episode_reward_fix,color='g', linewidth=1, linestyle='-',label='fixed')
        plt.plot(episode_reward_sac,color='b', linewidth=1, linestyle='-',label='SAC')
        # plt.plot(episode_reward_ddpg,color='r', linewidth=1, linestyle='-',label='DDPG')

        plt.xlabel('Episodes')
        plt.ylabel('Total reward')
        plt.legend()

        plt.savefig('./SAC/ddpg+sac.jpg')
        plt.savefig('./SAC/ddpg+sac.eps')
        plt.savefig('./SAC/ddpg+sac.pdf')
    if POLICY_FLAG:
        plt.figure(2,dpi=200)
        plt.plot(episode_reward_sto,color='y', linewidth=1, linestyle='-',label='stocastic')
        plt.plot(episode_reward_sac,color='b', linewidth=1, linestyle='-',label='SAC')
        plt.plot(episode_reward_sac_tra10,color='r', linewidth=1, linestyle='-',label='SAC_tra(I=10)')
        plt.plot(episode_reward_sac_tra15,color='g', linewidth=1, linestyle='-',label='SAC_tra(I=15)')

        plt.xlabel('Episodes')
        plt.ylabel('Total reward(kJ)')
        plt.legend()
        plt.savefig('./SAC/policy.jpg')
        plt.savefig('./SAC/policy.eps')
        plt.savefig('./SAC/policy.pdf')

    #DDPG and SAC Q_loss
    if Q_LOSS_FLAG:

        plt.figure(3)
        plt.plot(q,color='b',linewidth=1, linestyle='-',label='loss_SAC')
        plt.plot(td_error,color='r', linewidth=1, linestyle='-',label='loss_DDPG')
        
        plt.xlabel('Steps')
        plt.ylabel('the Q loss of SAC and DDPG')
        plt.legend()
        plt.savefig('./SAC/loss.jpg')

   
    if Q_VALUE_FLAG:
        #sac
        # plt.figure(2)
        # plt.plot(entropy)
        
        plt.figure(6)
        plt.plot(p)
        plt.figure(7)
        plt.plot(alpha)

        #DDPG
        plt.figure(8)
        plt.plot(a_loss)
    
    if DDPG_TRA_FLAG:

         #trajectory
        initial_location=0
        trajectory_length=100

        # l_uav_location_ddpg=l_uav_location_ddpg[initial_location:initial_location+trajectory_length]
        l_uav_location_ddpg=np.array(l_uav_location_ddpg)
        

        # f_uav_location_ddpg=f_uav_location_ddpg[initial_location:initial_location+trajectory_length]
        f_uav_location_ddpg=np.array(f_uav_location_ddpg)

    #DDPG trajectory
        plt.figure(9)
        ax1 = plt.axes(projection='3d')

        l_uav_location_ddpg_x=l_uav_location_ddpg[:,0]
        l_uav_location_ddpg_y=l_uav_location_ddpg[:,1]

        ax1.plot3D(l_uav_location_ddpg_x, l_uav_location_ddpg_y, 150, )

        for i in range(f_uav_num):
            f_uav_location_ddpg_x=f_uav_location_ddpg[:,i,0]
            f_uav_location_ddpg_y=f_uav_location_ddpg[:,i,1]
            ax1.plot3D(f_uav_location_ddpg_x, f_uav_location_ddpg_y, 140 )
        ax1.set_title('3D line plot')
        plt.savefig('./SAC/path_ddpg.eps')

    if SAC_TRA_FLAG:

      
    
        #for SAC
        # l_uav_location_sac=l_uav_location_sac[initial_location:initial_location+trajectory_length]
        l_uav_location_sac=np.array(l_uav_location_sac)     #.reshape(-1,2) #第0维
        

        # f_uav_location_sac=f_uav_location_sac[initial_location:initial_location+trajectory_length]
        f_uav_location_sac=np.array(f_uav_location_sac)  #.reshape(-1,f_uav_num,2) 

    #SAC trajectory

        plt.figure(10, dpi=200)
        ax = plt.axes(projection='3d')

        l_uav_location_sac_x=l_uav_location_sac[:,0]
        l_uav_location_sac_y=l_uav_location_sac[:,1]

        ax.plot3D(l_uav_location_sac_x, l_uav_location_sac_y, 150, label='The trajectory of H-UAV')
        
        
        tra = []
        for i in range(f_uav_num):
            f_uav_location_sac_x=f_uav_location_sac[:,i,0]
            f_uav_location_sac_y=f_uav_location_sac[:,i,1]
            # tra_, = ax.plot3D(f_uav_location_sac_x, f_uav_location_sac_y, 140,label = f'The trajectory of L-UAV{i}' )
            tra_, = ax.plot3D(f_uav_location_sac_x, f_uav_location_sac_y, 140 )
            tra.append(tra_)
        # ax.legend(tra, ['The trajectory of L-UAVs'])
        ax.legend()
        ax.set_title('3D line plot')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        plt.savefig('./SAC/path_sac.jpg')
        plt.savefig('./SAC/path_sac.eps')
        plt.savefig('./SAC/path_sac.pdf')


    if DISTANCE_FLAG:
        plt.figure(11,dpi=200)

        # plt.plot(d_sto,color='y', linewidth=1, linestyle='-',label='stocastic')
        # plt.plot(d_fix,color='g', linewidth=1, linestyle='-',label='fixed')
        plt.plot(d_sac,color='b', linewidth=1, linestyle='-',label='SAC(ε=0.001)')
        # plt.plot(d_ddpg,color='r', linewidth=1, linestyle='-',label='DDPG')

        plt.xlabel('Episodes')
        plt.ylabel('Maximum distance(m)')
        plt.legend()

        plt.savefig('./SAC/distance.jpg')
        plt.savefig('./SAC/distance.eps')
        plt.savefig('./SAC/distance.pdf')
    
    if TIME_FLAG :
        # plt.figure(4)
        # plt.plot(t_comm_ddpg,color='r',linewidth=1, linestyle='-',label='ddpg_t_comm')
        # plt.plot(t_total_ddpg,color='b',linewidth=1, linestyle='-',label='ddpg_t_total')
        # plt.legend()
        # plt.savefig('./SAC/ddpg.jpg')

        # plt.figure(5)
        # plt.plot(t_comm_sac,color='r',linewidth=1, linestyle='-',label='sac_t_comm')
        # plt.plot(t_total_sac,color='b',linewidth=1, linestyle='-',label='sac_t_total')
        # plt.legend()
        # plt.savefig('./SAC/sac.jpg')

        plt.figure(20)
        plt.plot(total_time_sac,color='r',linewidth=1, linestyle='-',label='total_time_sac')
        # plt.plot(t_total_sac,color='b',linewidth=1, linestyle='-',label='sac_t_total')
        plt.xlabel('Episodes')
        plt.ylabel('FL total time(s)')
        plt.legend()
    

        plt.savefig('./SAC/sac.jpg')
        plt.savefig('./SAC/sac.pdf')
        plt.savefig('./SAC/sac.eps')

    plt.show()

if __name__ == '__main__':
    draw()



    
    















