from robosuite_env.arena import Arena
from robosuite_mosaic.utils.mjcf_utils import xml_path_completion


class EmptyArena(Arena):
    """Empty workspace."""

    def __init__(self):
        super().__init__(xml_path_completion("empty_arena.xml"))
