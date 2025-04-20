import numpy as np

def make_prll_trgt(data, index, offset):
    """
    Shifts the nth column of a 2D numpy array by m values in the negative direction.

    Parameters:
        arr (numpy.ndarray): The input 2D array.
        n (int): The index of the column to shift.
        m (int): The number of positions to shift.

    Returns:
        numpy.ndarray: The shifted column as a new array.
    """
    if not (0 <= index < data.shape[1]):
        raise ValueError("Column index out of range.")
    
    trgt = np.roll(data[:, index], -offset)
    return data[:-offset], trgt[:-offset]