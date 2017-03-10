import numpy as np
import sys

# Setting the seed
seed = 7
np.random.seed(seed)

# Fish dictionary
FISH_DICTS = {
    "NoF":0,
    "ALB":1,
    "BET":2,
    "DOL":3,
    "LAG":4,
    "SHARK":5,
    "YFT":6,
    "OTHER":7    
}

print('Vars initialized (FISH_DICTS, seed)')