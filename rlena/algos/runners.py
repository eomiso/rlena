# TODO: implement RL algorithms

def coma(args):
    import os
    import gym
    from copy import deepcopy
    from pathlib import Path
    from easydict import EasyDict
    from rl2.agents.utils import LinearDecay
    from rlena.algos.agents.coma import COMAPGModel, COMAgent
    from rlena.algos.workers import ComaWorker
    from rlena.algos.utils import Logger
    args = EasyDict(args.__dict__)

    if args.env == 'pommerman':
        import pommerman
        from rlena.envs.customPomme import TwoVsTwoPomme

        env = TwoVsTwoPomme(ramdom_num_wall=args.random_num_wall,
                            max_rigid=args.max_rigid,
                            max_wood=args.max_wood,
                            max_steps=args.max_steps,
                            remove_stop=args.remove_stop,
                            onehot=args.onehot)
        env.seed(args.seed)
        _ = env.reset()

        # Setup shapes
        g_obs_shape = env.get_global_obs_shape()
        observation_shape = env.observation_shape
        ac_shape = (env.action_space.n - int(args.remove_stop),)
        # -1 in action_space.n to remove stop action

        config = args
        if config.mode == 'train':
            training = True
            max_episode = 20000
            render_mode = 'rgb_array'
            render_interval = 10
            save_gif = False
        else:
            training = False
            config.explore = False
            max_episode = 10
            render_mode = 'human'
            render_interval = 1
            save_gif = True

        # Epsilon decay
        eps = LinearDecay(start=args.eps_start,
                          end=args.eps_end,
                          decay_step=args.decay_step)

        logger = Logger(name=args.algo, args=config)

        config.log_dir = logger.log_dir
        config.ckpt_dir = os.path.join(logger.log_dir, 'ckpt')
        Path(config.ckpt_dir).mkdir(parents=True, exist_ok=True)

        team1 = COMAPGModel(observation_shape, g_obs_shape,
                            ac_shape, n_agents=2, **config)
        team2 = COMAPGModel(observation_shape, g_obs_shape,
                            ac_shape, n_agents=2, **config)

        magent = COMAgent([team1, team2], eps=eps, **config)

        worker = ComaWorker(
            env,
            n_env=args.n_env,
            agent=magent,
            n_agents=4,
            max_episodes=max_episode,
            training=training,
            logger=logger,
            log_interval=config.log_interval,
            render=True,
            render_mode=render_mode,
            render_interval=render_interval,
            save_gif=save_gif,
            is_save=True,
        )

    if args.env == 'snake':
        raise NotImplementedError

    worker.run()


def qmix(args):
    import os
    import yaml
    import argparse
    import numpy as np
    from tqdm import trange
    from copy import deepcopy
    from datetime import datetime
    
    import torch
    from torch.utils.tensorboard import SummaryWriter
    
    from rlena.algos.agent import QMIXAgent, QMIXCritic
    from rlena.algos.agents.baselines import StopedAgent, NoBombSimpleAgent
    args = EasyDict(args.__dict__)

    if args.env == 'pommerman':
        from rlena.envs.playground.pommerman.configs import team_v0_fast_env
        from rlena.envs.playground.pommerman import agents
        from rlena.envs.customPomme import OnehotEnvWrapper
        
        # config file open
        with open(args.config) as f:
            total_config = yaml.load(f, Loader=yaml.FullLoader)
            env_kwargs = total_config.pop('env_kwargs')
            agent_config = total_config.pop('agent_config')
            QMIX_config = total_config.pop('QMIX_config')
            train_config = total_config.pop("train_config")
        train_config['render'] = args.render
        train_config['mode'] = args.mode
        train_config['load_model'] = args.pretrained 
        if args.mode != 'train':
            train_config['load_model'] = True
            train_config['max_step'] = 10

        # GPU setup
        if torch.cuda.is_available():
            device = torch.device('cuda:%d'%(int(train_config['gpu'])))
            print("GPU using status: ", device)
        else:
            device = torch.device('cpu')
            print("CPU using")
        
        agent_config['device'] = device
        QMIX_config['device'] = device

        # tensorboard
        date = datetime.now()
        tensorboard_name = train_config['tensorboard_name']
        summary = SummaryWriter(os.path.join("./log", tensorboard_name, date.strftime("%Y%b%d_%H_%M_%S")))

        # making agents and critic
        agent1 = QMIXAgent(agent_config)
        agent2 = QMIXAgent(agent_config)
        critic = QMIXCritic(agents=(agent1, agent2), configs=QMIX_config)

        # making env
        env_config = team_v0_fast_env()
        env_config['env_kwargs'].update(env_kwargs)
        # Indicate whether training or not
        enemy_dict = {"simple" : agents.SimpleAgent,
                    "stoped" : StopedAgent,
                    "nobomb" : NoBombSimpleAgent}
        agent_list = [(True,agent1),
                    (False,enemy_dict[train_config['enemy']]()),
                    (True,agent2),
                    (False,enemy_dict[train_config['enemy']]())]
        env = OnehotEnvWrapper(env_config, agent_list=agent_list)

        if args.pretrained:
            print("pretrained agent and critic are used")
            agent1.load(1)
            agent2.load(2)
            critic.load()

        worker = QmixWorker(
            env= env,
            agent=[agent1, agent2],
            critic=critic,
            config=train_config,
            logger=summary
        )

        worker.run()