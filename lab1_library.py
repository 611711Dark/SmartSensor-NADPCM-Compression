
import numpy as np 
### initialize
from collections import namedtuple

# n_bits = 4;
# max_q = (2 ** (n_bits -1)) -1;
# min_q = -2 ** (n_bits -1);

# print(n_bits, min_q, max_q)
'''
def quantize(a, n_bits):
    max_q = (2 ** (n_bits - 1)) - 1
    min_q = -2 ** (n_bits - 1)
    q=int(np.round(a))
    q_clipped = max(min_q, min(max_q, q))
    return q_clipped
'''
def quantize(value, n_bits):
    try:
        max_val = 2 ** (n_bits - 1) - 1
        min_val = -2 ** (n_bits - 1)
        if not np.isfinite(value):
            return 0
        clipped = np.clip(value, min_val, max_val)
        return int(np.round(clipped))
    except:
        return 0
