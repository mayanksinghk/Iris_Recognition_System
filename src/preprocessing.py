import cv2
import numpy as np
import math
import random
from src.interfaces import BasePreprocessor

class IrisPreprocessor(BasePreprocessor):
    """
    Implements Iris Pre-processing for Compressed Sensing.
    Encompasses Gamma Correction, Canny Edge Detection, Hough Transform localization,
    and Daugman's Rubber Sheet Normalization.
    """

    def __init__(self, gamma: float = 1.2, polar_height: int = 64, polar_width: int = 512):
        self.gamma = gamma
        self.polar_height = polar_height
        self.polar_width = polar_width

    def _apply_gamma_correction(self, image: np.ndarray) -> np.ndarray:
        inv_gamma = 1.0 / self.gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def _safe_extract_circles(self, circles: np.ndarray) -> np.ndarray:
        """Safely flattens the varying dimensional outputs of OpenCV 4 HoughCircles."""
        if circles is None:
            return np.array([])
        if circles.ndim == 3:
            return circles[0, :]
        elif circles.ndim == 2:
            return circles
        elif circles.ndim == 1:
            return np.array([circles])
        return np.array([])

    def _find_pupil(self, image: np.ndarray) -> tuple:
        """
        Localizes the inner pupil boundary using an ensemble averaging approach.
        Applies morphological closing to eliminate specular reflections inside the pupil.
        """
        pupil_circles = []
        param1 = 200
        param2 = 120
        
        # Kernel designed to close the specular reflection holes inside the pupil mask
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        
        while param2 > 35 and len(pupil_circles) < 100:
            for mdn in [5, 7, 9]:
                for thrs in [30, 40, 50, 60, 70]:
                    # Median Blur to soften the image
                    median = cv2.medianBlur(image, 2 * mdn + 1)

                    # Inverse Threshold: Dark pupil becomes a white blob
                    _, thres = cv2.threshold(median, thrs, 255, cv2.THRESH_BINARY_INV)

                    # Morphological Close: Fills in the black holes caused by reflections
                    thres = cv2.morphologyEx(thres, cv2.MORPH_CLOSE, morph_kernel, iterations=2)

                    # Fill Contours to guarantee a solid shape
                    contours, _ = cv2.findContours(thres.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(thres, contours, -1, (255), -1)

                    # Canny Edges with slight dilation to connect broken boundary lines
                    edges = cv2.Canny(thres, 20, 100)
                    kernel = np.ones((3, 3), np.uint8)
                    edges = cv2.dilate(edges, kernel, iterations=1)
                    
                    ksize = 2 * random.randrange(3, 8) + 1
                    edges = cv2.GaussianBlur(edges, (ksize, ksize), 0)

                    # HoughCircles using strict kwargs and radius limits
                    circles = cv2.HoughCircles(
                        edges, cv2.HOUGH_GRADIENT, dp=1, minDist=10,
                        param1=param1, param2=param2, minRadius=15, maxRadius=120
                    )
                    
                    extracted_circles = self._safe_extract_circles(circles)
                    if extracted_circles.size > 0:
                        extracted_circles = np.round(extracted_circles).astype("int")
                        for c in extracted_circles:
                            if len(c) == 3:
                                pupil_circles.append(tuple(c))

            param2 -= 5

        if not pupil_circles:
            raise ValueError("Pupil boundary could not be localized.")

        # Calculate the mean circle to filter out anomalous detections
        mean_x = int(np.mean([c[0] for c in pupil_circles]))
        mean_y = int(np.mean([c[1] for c in pupil_circles]))
        mean_r = int(np.mean([c[2] for c in pupil_circles]))
        
        return (mean_x, mean_y, mean_r)

    def _find_ext_iris(self, image: np.ndarray, pupil_circle: tuple) -> tuple:
        """
        Localizes the outer iris boundary by searching for circles concentric to the pupil.
        """
        total_circles = []
        param2 = 120
        p_x, p_y, p_r = pupil_circle
        
        radius_range = int(math.ceil(p_r * 1.5))
        center_range = int(math.ceil(p_r * 0.5))
        
        while param2 > 40 and len(total_circles) < 50:
            for mdn in [9, 13, 17]:
                for thrs2 in [400, 480, 550]:
                    median = cv2.medianBlur(image, 2 * mdn + 1)

                    edges = cv2.Canny(median, 0, thrs2, apertureSize=5)
                    kernel = np.ones((3, 3), np.uint8)
                    edges = cv2.dilate(edges, kernel, iterations=1)
                    
                    ksize = 2 * random.randrange(5, 11) + 1
                    edges = cv2.GaussianBlur(edges, (ksize, ksize), 0)

                    circles = cv2.HoughCircles(
                        edges, cv2.HOUGH_GRADIENT, dp=1, minDist=10,
                        param1=200, param2=param2, minRadius=radius_range, maxRadius=280
                    )
                    
                    extracted_circles = self._safe_extract_circles(circles)
                    if extracted_circles.size > 0:
                        extracted_circles = np.round(extracted_circles).astype("int")
                        
                        for c in extracted_circles:
                            if len(c) != 3:
                                continue
                            c_x, c_y, c_r = c
                            # Filter based on concentricity to the previously found pupil
                            dist = math.sqrt((c_x - p_x)**2 + (c_y - p_y)**2)
                            if dist <= center_range and c_r > radius_range:
                                total_circles.append(tuple(c))
                                
            param2 -= 5

        if not total_circles:
            raise ValueError("Exterior iris boundary could not be localized.")

        mean_x = int(np.mean([c[0] for c in total_circles]))
        mean_y = int(np.mean([c[1] for c in total_circles]))
        mean_r = int(np.mean([c[2] for c in total_circles]))
        
        return (mean_x, mean_y, mean_r)

    def _daugman_normalization(self, image: np.ndarray, pupil_circle: tuple, iris_circle: tuple) -> np.ndarray:
        """
        Unwraps the annular iris region into a rectangular matrix.
        """
        normalized_iris = np.zeros((self.polar_height, self.polar_width), dtype=np.uint8)
        angles = np.linspace(0, 2 * np.pi, self.polar_width)
        
        p_x, p_y, p_r = pupil_circle
        i_x, i_y, i_r = iris_circle
        
        for i, theta in enumerate(angles):
            x_p = p_x + p_r * np.cos(theta)
            y_p = p_y + p_r * np.sin(theta)
            
            x_i = i_x + i_r * np.cos(theta)
            y_i = i_y + i_r * np.sin(theta)
            
            for j in range(self.polar_height):
                r = j / float(self.polar_height - 1)
                
                x = int((1 - r) * x_p + r * x_i)
                y = int((1 - r) * y_p + r * y_i)
                
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    normalized_iris[j, i] = image[y, x]
                    
        return normalized_iris

    def process(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        enhanced_image = self._apply_gamma_correction(image)
        
        pupil_circle = self._find_pupil(enhanced_image)
        iris_circle = self._find_ext_iris(enhanced_image, pupil_circle)
        
        normalized_iris = self._daugman_normalization(enhanced_image, pupil_circle, iris_circle)
        
        equalized_iris = cv2.equalizeHist(normalized_iris)
        
        return equalized_iris