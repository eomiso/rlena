import pommerman
from pommerman import agents
from algos import sac

def main(): 
    print(pommerman.REGISTRY)

    # Create a set of agents (exactly four)
    agent_list = [
        agents.SimpleAgent(),
        agents.RandomAgent(),
        agents.SimpleAgent(),
        agents.RandomAgent()
    ]

    env = pommerman.make('PommeFFACompetition-v0', agent_list)
    
    # Run the episodes just like OpenAI Gym
    for i_episode in range(1):
        state = env.reset()
        done = False
        while not done:
            env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
        print('Episode {} finished'.format(i_episode))
    env.close()
if __name__ == '__main__':
    main()


if __name__ == '__main__':
    main()