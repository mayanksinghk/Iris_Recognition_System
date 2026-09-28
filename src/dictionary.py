import os
import numpy as np
from typing import List, Tuple, Any
from src.interfaces import BaseDictionaryBuilder, BasePreprocessor, BaseFeatureExtractor
from src.utils import load_image

class DictionaryBuilder(BaseDictionaryBuilder):
    """
    Constructs the training dictionary matrix A for Compressed Sensing[cite: 1].
    Each column in the dictionary corresponds to a flattened feature vector of a training image.
    """

    def build(
        self, 
        preprocessor: BasePreprocessor, 
        extractor: BaseFeatureExtractor, 
        image_paths: List[str], 
        labels: List[Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Builds the dictionary matrix A and the corresponding label mapping.
        
        Args:
            preprocessor: Instantiated object implementing BasePreprocessor.
            extractor: Instantiated object implementing BaseFeatureExtractor.
            image_paths: List of file paths to the CASIA training images[cite: 1].
            labels: List of subject identities corresponding to each image path.
            
        Returns:
            Tuple containing the 2D dictionary matrix A and the 1D label array.
        """
        if len(image_paths) != len(labels):
            raise ValueError("The number of image paths must match the number of labels.")

        feature_vectors = []
        valid_labels = []

        for path, label in zip(image_paths, labels):
            if not os.path.exists(path):
                print(f"Warning: File not found {path}. Skipping.")
                continue
                
            try:
                # Load grayscale image[cite: 2]
                image = load_image(path, show=False)
                
                # Pre-process to get normalized iris[cite: 1]
                normalized_iris = preprocessor.process(image)
                
                # Extract 1D feature vector[cite: 1]
                feature_vector = extractor.extract(normalized_iris)
                
                feature_vectors.append(feature_vector)
                valid_labels.append(label)
                
            except Exception as e:
                print(f"Error processing {path}: {e}. Skipping.")

        if not feature_vectors:
            raise RuntimeError("No valid feature vectors were extracted. Dictionary is empty.")

        # Horizontally stack vectors to create matrix A (Dimensions: Feature_Length x Number_of_Samples)
        dictionary_A = np.column_stack(feature_vectors)
        labels_array = np.array(valid_labels)

        return dictionary_A, labels_array