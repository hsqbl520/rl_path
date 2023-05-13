import os
import time

import argparse
import numpy as np
from simple_model import MAModel
from simple_agent import MAAgent
from maddpg import MADDPG
from make_env import make_env
import paddle

#调用的是parl模型的API
from parl.utils import logger, summary
from gym import spaces

CRITIC_LR = 0.01 #
ACTOR_LR = 0.01  #
GAMMA = 0.95  #初始0.95#预测229步
TAU = 0.01  # soft update
BATCH_SIZE = 128

MAX_STEP_PER_EPISODE = 50 # maximum step per episode
EVAL_EPISODES = 3

# Runs policy and returns episodes' rewards and steps for evaluation
def run_evaluate_episodes(env, agents, eval_episodes):
    eval_episode_rewards = []
    eval_episode_steps = []
    while len(eval_episode_rewards) < eval_episodes:
        obs_n = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done and steps < MAX_STEP_PER_EPISODE:
            steps += 1
            action_n = [
                agent.predict(obs) for agent, obs in zip(agents, obs_n)
            ]
            obs_n, reward_n, done_n, _ = env.step(action_n)
            
            #done_n里全部为turn,函数才停止.
            done = all(done_n)
            total_reward += sum(reward_n)
            # show animation
            if args.show:
                time.sleep(0.1)
                env.render()

        eval_episode_rewards.append(total_reward)
        eval_episode_steps.append(steps)
    return eval_episode_rewards, eval_episode_steps

def evaluate_main():
    logger.set_dir('{}/evaluate_log/{}'.format(args.data_dir,args.date))
    if args.continuous_actions:
        assert isinstance(env.action_space[0], spaces.Box)
    critic_in_dim = sum(env.obs_shape_n) + sum(env.act_shape_n)
    # build agents
    agents = []
    for i in range(env.n):
        model = MAModel(env.obs_shape_n[i], env.act_shape_n[i], critic_in_dim,args.continuous_actions)
        algorithm = MADDPG(
            model,
            agent_index=i,
            act_space=env.action_space,
            gamma=GAMMA,
            tau=TAU,
            critic_lr=CRITIC_LR,
            actor_lr=ACTOR_LR)
        agent = MAAgent(
            algorithm,
            agent_index=i,
            obs_dim_n=env.obs_shape_n,
            act_dim_n=env.act_shape_n,
            batch_size=BATCH_SIZE)
        agents.append(agent)

    #2.载入训练完成的模型    
    for i in range(len(agents)):
        model_file = args.data_dir+"/models/" +args.restore_model_dir + '/agent_' + str(i)
        if not os.path.exists(model_file):
            raise Exception(
                'model file {} does not exits'.format(model_file))
        agents[i].restore(model_file)

    #3开始一个episodes        
    eval_episode_rewards, eval_episode_steps = run_evaluate_episodes(env, agents, EVAL_EPISODES)

    # summary.add_scalar('eval/episode_reward',np.mean(eval_episode_rewards), eval_episode_steps)
    # logger.info('Evaluation over: {} episodes, Reward: {}'.format(EVAL_EPISODES, np.mean(eval_episode_rewards)))

def run_episode(env, agents):
    obs_n = env.reset()
    #boundary_list=env.set_boundaries()
    done = False
    total_reward = 0
    #初始设置时,每个智能体的奖励设置为0.
    agents_reward = [0 for _ in range(env.n)]
    steps = 0
    while not done and steps < MAX_STEP_PER_EPISODE:
        steps += 1
        action_n = [agent.sample(obs,args.use_target_model) for agent, obs in zip(agents, obs_n)]
        #如果修改中途结束,会在这不卡输出错误动作
        next_obs_n, reward_n, done_n, _ = env.step(action_n)
        #done_n里全部为turn,函数才停止
        done = all(done_n)
            
        #这个地方加入判断,是否可行,不行然后重新采样.
        # store experience
        for i, agent in enumerate(agents):
            agent.add_experience(obs_n[i], action_n[i], reward_n[i],
                                 next_obs_n[i], done_n[i])

        # compute reward of every agent
        obs_n = next_obs_n
        for i, reward in enumerate(reward_n):
            total_reward += reward
            agents_reward[i] += reward

        # show model effect without training
        if args.restore and args.show:
            continue

        # learn policy
        for i, agent in enumerate(agents):
            critic_loss = agent.learn(agents)

    return total_reward, agents_reward, steps,done

def train_main():
    #基于paddle
    paddle.seed(args.seed)
    #np.random.seed(args.seed)
    #日志存储修改
    logger.set_dir('{}/train_log/{}'.format(args.data_dir,args.date))

    if args.continuous_actions:
        assert isinstance(env.action_space[0], spaces.Box)
    critic_in_dim = sum(env.obs_shape_n) + sum(env.act_shape_n)

    # build agents
    agents = []
    #env.n 应该就是智能体的数量
    for i in range(env.n):
        #1.建立神经网络模型
        model = MAModel(env.obs_shape_n[i], env.act_shape_n[i], critic_in_dim,
                        args.continuous_actions)
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

    #可选4.如果有模型用了，将权重直接放到神经网络里边去
    if args.restore:
        # restore modle
        for i in range(len(agents)):

            #zzh 修改
            model_file = args.data_dir+"/models/"+args.restore_model_dir + '/agent_' + str(i)

            if not os.path.exists(model_file):
                raise Exception(
                    'model file {} does not exits'.format(model_file))
            agents[i].restore(model_file)

    #5更新神经网络参数
    total_steps = 0
    total_episodes = 0
    #在最大episodes之下就继续进行训练
    while total_episodes <= args.max_episodes:
        # run an episode
        ep_reward, ep_agent_rewards, steps ,done= run_episode(env, agents)##进行一局训练
        summary.add_scalar('train/episode_reward_wrt_episode', ep_reward,total_episodes)
        summary.add_scalar('train/episode_reward_wrt_step', ep_reward, total_steps)
        #这个是日志打印保持的地方
        # if total_episodes%1000==0:
        logger.info(
            'episode {}, reward {}, agents rewards {}, episode steps {},done {}'
            .format(total_episodes, ep_reward, ep_agent_rewards,steps,done))

        total_steps += steps
        total_episodes += 1

        # evaluste agents
        if total_episodes % args.test_every_episodes == 0:

            eval_episode_rewards, eval_episode_steps = run_evaluate_episodes(env, agents, EVAL_EPISODES)
            summary.add_scalar('eval/episode_reward',np.mean(eval_episode_rewards), total_episodes)
            logger.info('Evaluation over: {} episodes, Reward: {}'.format(EVAL_EPISODES, np.mean(eval_episode_rewards)))

        # save model
        if total_episodes % args.auto_model_save_frequency== 0:
            #加入logger语句
            if args.store_model:
                episodes_son_dir="/"+str(total_episodes)
                
                model_dir =args.data_dir+ "/models/"+args.date +args.model_dir+episodes_son_dir
                print(model_dir)

                os.makedirs(os.path.dirname(model_dir), exist_ok=True)
                for i in range(len(agents)):
                    model_name = '/agent_' + str(i)
                    agents[i].save(model_dir + model_name)
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Environment
    parser.add_argument('--env',type=str,default='simple_spread',help='scenario of MultiAgentEnv')
    # auto save model, optional restore model
    parser.add_argument('--show', action='store_true', default=True, help='display or not')
    #save_path
    parser.add_argument('--data_dir',type=str,default='./data',help='date')
    #经常需要修改的参数
    parser.add_argument('--seed',type=int,default=2,help="It's seed!")

    parser.add_argument('--date',type=str,default='2023_4_5',help='date')

    # 测试
    parser.add_argument('--test_every_episodes',type=int,default=int(100), help='the episode interval between two consecutive evaluations')

    #载入模型及路径
    #载入模型自己写事件
    parser.add_argument('--restore',action='store_true',default=False,help='restore model or not, must have model_dir')

    parser.add_argument('--restore_model_dir',type=str,default='2023_4_5/processCpu_4_ChangeRatioAs7/8000',help='directory for saving model')
    
    #保存模型及主文件夹路径及自动保存间隔次数    
    parser.add_argument('--store_model',action='store_true',default=True,help='store model or not, must have model_dir')

    parser.add_argument('--model_dir',type=str,default='/processCpu_4_ChangeRatioAs7',help='directory for saving model')

    parser.add_argument('--auto_model_save_frequency',type=int,default=int(1e3),help='the episode(frequency) to auto sace model file ')
    
    #其他参数
    parser.add_argument('--continuous_actions',action='store_true',default=True,help='use continuous action mode or not')
    parser.add_argument('--max_episodes',type=int,default=150000,help='stop condition: number of episodes')
    parser.add_argument('--use_target_model', type=bool, default=True, help='use_target_model')
    args = parser.parse_args()
    env = make_env(args.env, args.continuous_actions)
    train_main()
    #evaluate_main()
