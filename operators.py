import numpy as np

def Exp_map(vec):
    """
    Returns Exponential Map of 3 * 1 Vector --> SO3 Rotational Matrix

    """
    mag = np.linalg.norm(vec)
    S = skew(vec)

    if mag < 1e-6:
        R = np.eye(3) + S
    else:
        one_minus_cos = 2 * np.sin(mag/2) * np.sin(mag/2)
        R = np.eye(3) + np.sin(mag)/mag * S + one_minus_cos/mag**2 * S**2

    return R

def Log_map(R):
    """
    Returns Logarithm Map of S03 Rotational Matrix --> 3 * 1 Vector
    
    """

def RightJac(vec):
    """
    Returns Right Jacobian of 3 * 1  Vector

    """
    mag = np.linalg.norm(vec)
    S = skew(vec)
    
    if mag < 1e-6:
        R = np.eye(3) - 1/2 * S
    else:
        one_minus_cos = 2 * np.sin(mag/2) * np.sin(mag/2)
        R = np.eye(3) - one_minus_cos/mag**2 * S + (1/mag**2 - np.sin(mag)/mag**3) * S**2
    
    return R

def skew(vec):
    """
    Returns skew symmetric form of 3 * 1 Vector

    """
    if len(vec) != 3:
        raise ValueError('Input to this function should be only 3 * 1 Vector Form')
    else:
        S = np.array([[0, -vec[2], vec[1]],
                    [vec[2], 0, -vec[0]],
                    [-vec[1], vec[0], 0]])
        return S

def vert(vec):
    """
    Converts shape (3,) numpy arrays to (3,1) array

    """
    if len(vec) != 3:
        raise ValueError('Input to this function should be only 3 * 1 Vector Form')
    else:
        res = np.zeros((3,1))
        res[0] = vec[0]
        res[1] = vec[1]
        res[2] = vec[2]