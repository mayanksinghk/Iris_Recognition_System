import numpy as np
from scipy.optimize import linprog
from src.interfaces import BaseSparseSolver

class BasisPursuitSolver(BaseSparseSolver):
    """
    Finds the sparse vector using Basis Pursuit[cite: 1].
    Solves the L1 minimization problem: min ||x||_1 subject to Ax = y.
    
    Frames the L1 minimization as a standard Linear Programming problem to ensure 
    compatibility with SciPy's linprog optimizer.
    """

    def solve(self, dictionary: np.ndarray, test_vector: np.ndarray) -> np.ndarray:
        """
        Calculates the sparse representation of the test vector.
        
        Args:
            dictionary (np.ndarray): The training dictionary matrix A (Dimensions: Feature_Length x Samples).
            test_vector (np.ndarray): The feature vector y of the test image.
            
        Returns:
            np.ndarray: The sparse coefficient vector x.
        """
        # Ensure the test vector is a 1D array
        y = test_vector.flatten()
        
        # Dimensions: m (feature length), n (number of dictionary samples)
        m, n = dictionary.shape
        
        # To minimize the L1 norm ||x||_1, we split x into positive and negative parts:
        # x = u - v, where u >= 0 and v >= 0.
        # This converts the non-linear absolute value into a linear objective: min sum(u) + sum(v)
        
        # Objective function coefficients (c): Vector of ones of size 2n for [u; v]
        c = np.ones(2 * n)
        
        # Equality constraint matrix (A_eq): [A, -A] since Ax = A(u - v) = Au - Av = y
        A_eq = np.hstack((dictionary, -dictionary))
        
        # Bounds for u and v (must be non-negative)
        bounds = [(0, None) for _ in range(2 * n)]
        
        # Solve the linear programming problem
        # Highs-ds (dual simplex) is typically faster and more robust for Basis Pursuit
        result = linprog(c, A_eq=A_eq, b_eq=y, bounds=bounds, method='highs-ds')
        
        if not result.success:
            print(f"Warning: Basis pursuit optimization failed: {result.message}")
            # Return a zero vector as a fallback if optimization fails
            return np.zeros(n)
            
        # Reconstruct the original sparse vector x from u and v
        u = result.x[:n]
        v = result.x[n:]
        sparse_x = u - v
        
        return sparse_x