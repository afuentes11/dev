import numpy as np

def zip_shuffled(*arrays):
    """
    Shuffle multiple arrays in unison and return them as a list of tuples.
    
    Parameters:
    -----------
    *arrays : array-like
        One or more arrays to be shuffled. Each array must have the same length.
        
    Returns:
    --------
    list of tuples
        A list where each tuple contains elements from the input arrays at the same index, but in shuffled order.
    
    Raises:
    -------
    ValueError
        If the input arrays do not have the same length.
    
    Examples:
    ---------
    >>> import numpy as np
    >>> a = np.array([1, 2, 3, 4, 5])
    >>> b = np.array([10, 20, 30, 40, 50])
    >>> zip_shuffled(a, b)
    [(3, 30), (1, 10), (5, 50), (2, 20), (4, 40)]
    
    Notes:
    ------
    - This function is useful for shuffling datasets consisting of multiple features or labels.
    - The input arrays are not modified in place; instead, a shuffled list of tuples is returned.
    - The function uses `numpy.random.shuffle` to shuffle the arrays.
    """
    if not all(len(arr) == len(arrays[0]) for arr in arrays):
        raise ValueError("All input arrays must have the same length.")
    
    arrays = [*zip(*arrays)]
    np.random.shuffle(arrays)
    return arrays