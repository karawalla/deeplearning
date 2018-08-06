import numpy as np
import pandas as pd


def init_weight_and_bias(M1, M2):
    W = np.random.randn(M1, M2) / np.sqrt(M1 * M2)
    b = np.zeros(M2)
    return W.astype(np.float32), b.astype(np.float32)
