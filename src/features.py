import cv2
import numpy as np
from src.interfaces import BaseFeatureExtractor

class GaborFeatureExtractor(BaseFeatureExtractor):
    """
    Replaces the spatial SIFT matching from the original implementation[cite: 2] 
    with a Gabor filter bank suitable for Compressed Sensing[cite: 1].
    Extracts textural frequencies and flattens them into a 1D column vector.
    """

    def __init__(self, ksize: int = 15, target_size: tuple = (64, 16)):
        """
        Args:
            ksize (int): Size of the Gabor kernel.
            target_size (tuple): Width and height to downsample the iris before 
                                 flattening. In Compressed Sensing, flattening a 
                                 full 512x64 image across multiple filters creates 
                                 a vector too large for efficient L1 optimization.
        """
        self.ksize = ksize
        self.target_size = target_size
        self.filters = self._build_filter_bank()

    def _build_filter_bank(self) -> list:
        """
        Constructs a bank of Gabor filters across multiple orientations and scales
        to capture the complex ridge patterns of the iris.
        """
        filters = []
        # Use 2 scales (frequencies) and 4 orientations (thetas)
        for theta in np.arange(0, np.pi, np.pi / 4):
            for lamda in [np.pi / 4, np.pi / 2]: 
                kern = cv2.getGaborKernel(
                    ksize=(self.ksize, self.ksize), 
                    sigma=3.0, 
                    theta=theta, 
                    lambd=lamda, 
                    gamma=0.5, 
                    psi=0, 
                    ktype=cv2.CV_32F
                )
                kern /= 1.5 * kern.sum()  # Normalize to prevent extreme scaling
                filters.append(kern)
        return filters

    def extract(self, normalized_iris: np.ndarray) -> np.ndarray:
        """
        Applies the Gabor filter bank to the unwrapped iris, pools the results, 
        and flattens them into the 1D sparse-ready vector[cite: 1].
        
        Args:
            normalized_iris (np.ndarray): The 2D unwrapped rectangular iris.
            
        Returns:
            np.ndarray: A 1D numpy array representing the feature vector (y).
        """
        # Downsample the normalized iris to reduce dictionary dimensionality
        resized_iris = cv2.resize(normalized_iris, self.target_size, interpolation=cv2.INTER_AREA)
        
        filter_responses = []
        for kern in self.filters:
            # Apply each Gabor filter to the downsampled image
            fmap = cv2.filter2D(resized_iris, cv2.CV_8UC3, kern)
            filter_responses.append(fmap)
            
        # Stack all filtered responses and flatten into a single 1D column vector
        feature_vector = np.array(filter_responses).flatten()
        
        # Normalize the final vector to ensure stable L1 minimization in Basis Pursuit
        vector_norm = np.linalg.norm(feature_vector)
        if vector_norm > 0:
            feature_vector = feature_vector / vector_norm
            
        return feature_vector