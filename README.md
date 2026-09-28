# Iris Recognition System
This repository provides a complete, Python-based biometric identification pipeline for iris recognition. Contained entirely within `iris.py`, the system leverages OpenCV and classical computer vision techniques to segment the iris, extract robust features, and verify identity by comparing local scale-invariant descriptors.

## Core Capabilities
- **Accurate Boundary Detection**: Utilizes Canny edge detection, median blurring, and custom Hough Circle Transform heuristics to isolate the pupil (inner boundary) and the exterior iris (outer boundary).
- **Image Normalization**: Applies intelligent masking to remove eyelids and eyelashes (specifically targeting the top sector between 50 and 130 degrees) and uses histogram equalization to standardize lighting.
- **ROI Segmentation**: Unwraps and isolates the iris into discrete geometric patches: `right-side`, `left-side`, `bottom`, and `complete` for localized feature analysis.
- **SIFT Feature Extraction**: Computes Scale-Invariant Feature Transform (SIFT) keypoints and descriptors within the defined iris regions, actively filtering out noisy keypoints that fall inside the pupil or outside the iris bounds.
- **Robust Feature Matching**: Matches geometric keypoints between two images using a Brute-Force Matcher (KNN). The matches are rigorously filtered using a spatial distance ratio, as well as angle and distance standard deviation checks to eliminate false positives.
- **Binary Serialization**: Supports storing and loading pre-computed keypoints and descriptors using `gzip` and `pickle` to drastically speed up repetitive comparisons.

## Prerequisites
To run the recognition pipeline, you will need a Python 3 environment with the following dependencies:
- `numpy`
- `opencv-python` (`cv2`) - Note: SIFT is utilized, requiring OpenCV version 4.4.0 or higher.
- `matplotlib`

Standard library modules used include `os`, `sys`, `math`, `random`, `pickle`, `copy`, `gzip`, `inspect`, and `itertools`.
## File Structure
The repository relies on a single modular script:

| **File**  | **Description**                                                                                                         |
| --------- | ----------------------------------------------------------------------------------------------------------------------- |
| `iris.py` | The main execution script containing the full computer vision pipeline, from boundary segmentation to feature matching. |

## Usage
### Direct Execution
By default, executing `iris.py` runs a localized comparison test on two provided sample images. Ensure the sample images (`IMG_002_L_2.JPG` and `IMG_002_R_3.JPG`) are located in the same working directory.

```bash
python iris.py
```

### Programmatic Integration

You can import `iris.py` into other systems or evaluation scripts to utilize specific stages of the pipeline:

Python

```python
import iris

# Compare two raw iris images directly
iris.compare_images('path/to/image1.JPG', 'path/to/image2.JPG')

# Compare two serialized/pickled ROI binaries
iris.compare_binfiles('path/to/data1.gz', 'path/to/data2.gz')
```

## Pipeline Architecture
1. **Preprocessing (`load_image`)**: Loads the input images in grayscale mode.
2. **Segmentation (`get_iris_boundaries`, `find_pupil`, `find_ext_iris`)**: Iteratively scans for circular gradients to isolate the distinct inner (pupil) and outer (sclera) boundaries.
3. **Enhancement (`get_equalized_iris`)**: Masks noisy physiological regions and equalizes the pixel intensity distribution.
4. **Feature Mapping (`get_rois`, `load_keypoints`, `load_descriptors`)**: Flattens regions of interest and computes SIFT spatial features.
5. **Matching & Verification (`getall_matches`, `get_matches`)**: Computes k-NN matches and applies rigid geometric validation (median differences and standard deviation thresholds) to return a quantitative similarity assessment.