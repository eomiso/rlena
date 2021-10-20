from rlena.envs.playground.pommerman import characters
from rlena.envs.playground.pommerman.agents import BaseAgent, SimpleAgent

class StopedAgent(BaseAgent):
    def __init__(self, character=characters.Bomber):
        super(StopedAgent, self).__init__(character)

    def act(self, obs, action_space):
        return 0 # stop action

class NoBombSimpleAgent(SimpleAgent):
    def act(self, obs, action_space):
        action = super().act(obs, action_space)
        if action == 5: # Bomb action
            action = np.random.choice(5)
        return action