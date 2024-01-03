import numpy as np
from dezero import cuda
from dezero.core import Function, as_variable
from dezero.utils import pair, get_conv_outsize, get_deconv_outsize
from dezero.functions import linear, broadcast_to, im2col

def conv2d_simple(x, W, b=None, stride=1, pad=0):
    x, W = as_variable(x), as_variable(W)

    Weight = W # to distinguish it with width(W)
    N, C, H, W = x.shape
    OC, C, KH, KW = Weight.shape
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    col = im2col(x, (KH, KW), stride, pad, to_matrix=True) # (N*OH*OW) x (C*KH*KW)
    Weight = Weight.reshape(OC, -1).transpose() # OC x (C*KH*KW) -> trans: (C*KH*KW) x OC
    t = linear(col, Weight, b) # t.shape = (N*OH*OW) x OC
    y = t.reshape(N, OH, OW, OC).transpose(0, 3, 1, 2) # y.shape - N x OC x OH x OW
    return y


def pooling_simple(x, kernel_size, stride=1, pad=0):
    x = as_variable(x)

    N, C, H, W = x.shape
    KH, KW = pair(kernel_size)
    PH, PW = pair(pad)
    SH, SW = pair(stride)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_deconv_outsize(W, KW, SW, PW)

    col = im2col(x, kernel_size, stride, pad, to_matrix=True)
    col.reshape(-1, KH*KW) # (N*OH*OW*C) x (KH*KW)
    y = col.max(axis=1) # y.shape - (N*OH*OW*C, )
    y = y.reshape(N, OH, OW, C).transpose(0, 3, 1, 2)
    return y