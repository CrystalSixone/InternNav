from internnav.agent.base import Agent
from internnav.configs.agent import AgentCfg


@Agent.register('seq2seq_clip')
class Seq2SeqCLIPAgent(Agent.agents['cma_clip']):
    def __init__(self, config: AgentCfg):
        super().__init__(config)
