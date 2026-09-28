import numpy as np
from typing import Any
from src.interfaces import BaseMatcher

class ResidualMatcher(BaseMatcher):
    """
    Implements classification by matching the sparse vector against dictionary classes[cite: 1].
    The assigned identity is the class that yields the minimal reconstruction residual.
    """

    def match(
        self, 
        dictionary: np.ndarray, 
        test_vector: np.ndarray, 
        sparse_vector: np.ndarray, 
        labels: np.ndarray
    ) -> Any:
        """
        Computes the residual r_i = ||y - A_i x_i||_2 for each distinct class i.
        
        Args:
            dictionary (np.ndarray): The training dictionary matrix A.
            test_vector (np.ndarray): The original feature vector y of the test image.
            sparse_vector (np.ndarray): The computed sparse vector x.
            labels (np.ndarray): The array of subject labels corresponding to dictionary columns.
            
        Returns:
            Any: The label of the subject with the minimum reconstruction error.
        """
        unique_classes = np.unique(labels)
        min_residual = float('inf')
        best_match_label = None
        
        # y: ground truth test vector
        y = test_vector.flatten()
        
        for cls in unique_classes:
            # Find column indices belonging to the current class
            class_indices = np.where(labels == cls)[0]
            
            # Isolate A_i: only the columns of A belonging to class i
            A_i = dictionary[:, class_indices]
            
            # Isolate x_i: only the coefficients of x belonging to class i
            x_i = sparse_vector[class_indices]
            
            # Reconstruct the test vector using only the current class's components
            y_hat = A_i.dot(x_i)
            
            # Calculate the L2 norm (Euclidean distance) of the residual
            residual = np.linalg.norm(y - y_hat)
            
            if residual < min_residual:
                min_residual = residual
                best_match_label = cls
                
        if best_match_label is None:
            raise ValueError("Matching failed: No valid classes evaluated.")
            
        return best_match_label