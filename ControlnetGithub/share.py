from ControlnetGithub import config
from ControlnetGithub.cldm.hack import disable_verbosity, enable_sliced_attention

disable_verbosity()

if config.save_memory:
    enable_sliced_attention()
