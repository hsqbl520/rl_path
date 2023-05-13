import numpy as np
from simple_model import MAModel
from simple_agent import MAAgent
from parl.algorithms import MADDPG
from make_env import make_env
import paddle

CRITIC_LR = 0.01 #
ACTOR_LR = 0.01  #
GAMMA = 0.95  #初始0.95#预测229步
TAU = 0.01  # soft update
BATCH_SIZE = 1024#和深度的batchsize不是一种东西
MAX_STEP_PER_EPISODE = 25 # maximum step per episode
EVAL_EPISODES = 3

env = make_env('simple_spread', True)

obs_n = env.reset()
print(env.act_shape_n)
if __name__ == '__main__':
    critic_in_dim = 4
    #critic_in_dim = sum(env.obs_shape_n) + sum(env.act_shape_n)

    # build agents
    agents = []
    #env.n 应该就是智能体的数量
    for i in range(env.n):
        #1.建立神经网络模型
        model = MAModel(env.obs_shape_n[i], env.act_shape_n[i], critic_in_dim,
                        True)
        #2.算法，后期再改
        algorithm = MADDPG(
            model,
            agent_index=i,
            act_space=env.action_space,
            gamma=GAMMA,
            tau=TAU,
            critic_lr=CRITIC_LR,
            actor_lr=ACTOR_LR)
        #3.创建 agent
        agent = MAAgent(
            algorithm,
            agent_index=i,
            obs_dim_n=env.obs_shape_n,
            act_dim_n=env.act_shape_n,
            batch_size=BATCH_SIZE)
        #4所有的agent得到一个列表
        agents.append(agent)
        print(model.value(np.array([0,0,0,0])))
