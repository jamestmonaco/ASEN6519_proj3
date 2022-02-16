'''
mls.py

Routine for generating maximum length sequences

@author Brian Breitsch
@email brian.breitsch@colorado.edu
'''

import numpy
from numpy import zeros, ones, int8

def generate_mls(N, feedback_taps, output_taps):
    '''Generates Maximum Length Sequence (length-(2**N - 1) binary sequence) for the given feedback and output taps.
    
    Parameters
    ----------
    `N` : int
        number of bits to use in the feedback register
    `feedback_taps` : array or ndarray of shape (L,)
        the taps to use for feedback to the shift register's first value
    `output_taps` : array or ndarray of shape (M,)
        the taps to use for choosing the code output

    Returns
    -------
    output : ndarray of shape(2**N - 1,)
        the code sequence
    '''
    shift_register = ones((N,), dtype=int8)
    seq = zeros((2**N - 1,), dtype=int8)
    for i in range(2**N - 1):
        seq[i] = numpy.sum(shift_register[output_taps]) % 2
        first = numpy.sum(shift_register[feedback_taps]) % 2
        shift_register[1:] = shift_register[:-1]
        shift_register[0] = first
    return seq