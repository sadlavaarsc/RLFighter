import numpy as np

from rlfighter.agents.base import Controller


class RLController(Controller):
    """Wrapper for a policy network."""

    def __init__(self, policy) -> None:
        self.policy = policy
        self.deterministic = False

    def act(self, agent_id: int, world) -> tuple[int, int]:
        # This is a stub; the actual training loop will call the policy directly
        # with observations and return actions.
        # For eval/play, we'll override this later.
        return (0, 8)
