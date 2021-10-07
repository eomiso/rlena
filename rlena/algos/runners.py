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
