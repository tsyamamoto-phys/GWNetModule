"""
_utils.py
"""
import math

def _cal_length(N, k, s=1, d=1, printflg=False):
    
    ret = math.floor((N - d*(k-1) -1) / s + 1)
    if printflg: print("output length: ", ret)
    return ret


def _cal_length4deconv(N, k, s=1, pad=0, outpad=0, d=1, printflg=False):

    ret = (N-1) * s - 2 * pad + d * (k-1) + outpad + 1
    if printflg: print("output length: ", ret)
    return ret

def _cal_length4upsample(N, scale, printflg=False):

    ret = N * scale
    if printflg: print("output length: ", ret)
    return ret