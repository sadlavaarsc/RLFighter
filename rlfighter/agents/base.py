from abc import ABC, abstractmethod


class Controller(ABC):
    @abstractmethod
    def act(self, agent_id: int, world) -> tuple[int, int]:
        """Return (action_type, move_dir) for the given agent."""
        ...
