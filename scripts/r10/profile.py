import os,json
import torch
from dpa_ctta.r10_use_write_rl.profiling import Profile
torch.set_num_threads(2)
Profile(json.load(open(os.environ['R10_CONFIG']))).run()
