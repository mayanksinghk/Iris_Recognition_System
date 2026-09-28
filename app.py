import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

# Import the concrete implementations from the src module
from src.preprocessing import IrisPreprocessor
from src.features import GaborFeatureExtractor
from src.dictionary import DictionaryBuilder
from src.sparse_solver import BasisPursuitSolver
from src.matching import ResidualMatcher

def parse_dataset_directory(dataset_path: str) -> tuple[list[str], list[str]]:
    """
    Parses the nested dataset directory structure: dataset/<ID>/<L or R>/<image>
    Returns a list of file paths and a corresponding list of identity labels.
    """
    image_paths = []
    labels = []
    
    if not os.path.exists(dataset_path):
        return image_paths, labels

    # Iterate through numbered subject folders
    for subject_id in os.listdir(dataset_path):
        subject_dir = os.path.join(dataset_path, subject_id)
        if not os.path.isdir(subject_dir):
            continue
            
        # Iterate through Left (L) and Right (R) eye folders
        for eye_side in ['L', 'R']:
            eye_dir = os.path.join(subject_dir, eye_side)
            if not os.path.isdir(eye_dir):
                continue
                
            # Iterate through images
            for img_name in os.listdir(eye_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(eye_dir, img_name)
                    image_paths.append(img_path)
                    
                    # Create a unique label, e.g., "001_L"
                    labels.append(f"Subject_{subject_id}_{eye_side}")
                    
    return image_paths, labels

@st.cache_data(show_spinner=False)
def load_and_build_dictionary(dataset_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Caches the computationally expensive dictionary building process.
    Only re-runs if the dataset_path string changes.
    """
    image_paths, labels = parse_dataset_directory(dataset_path)
    
    if not image_paths:
        raise FileNotFoundError(f"No valid images found in {dataset_path}. Check the directory structure.")
        
    preprocessor = IrisPreprocessor(gamma=1.2)
    extractor = GaborFeatureExtractor()
    builder = DictionaryBuilder()
    
    dictionary_A, labels_array = builder.build(preprocessor, extractor, image_paths, labels)
    return dictionary_A, labels_array

def process_uploaded_image(uploaded_file) -> np.ndarray:
    """Converts a Streamlit UploadedFile object into an OpenCV grayscale image."""
    image_pil = Image.open(uploaded_file).convert('L')
    return np.array(image_pil)

def main():
    st.set_page_config(page_title="CS Iris Recognition", layout="wide")
    st.title("Compressed Sensing Iris Recognition")
    
    # Initialize pipeline components
    preprocessor = IrisPreprocessor(gamma=1.2)
    extractor = GaborFeatureExtractor()
    solver = BasisPursuitSolver()
    matcher = ResidualMatcher()
    
    # Sidebar for Configuration & Dictionary Building
    st.sidebar.header("System Configuration")
    dataset_path = st.sidebar.text_input("Dataset Directory Path", value="./dataset")
    
    if "dictionary_A" not in st.session_state:
        st.session_state.dictionary_A = None
        st.session_state.labels_array = None

    if st.sidebar.button("Build / Load Dictionary"):
        with st.spinner("Building dictionary matrix A... This may take a while."):
            try:
                A, labels = load_and_build_dictionary(dataset_path)
                st.session_state.dictionary_A = A
                st.session_state.labels_array = labels
                st.sidebar.success(f"Dictionary Built! Shape: {A.shape}")
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")

    # Main Area for Testing
    st.header("Test Image Identification")
    
    if st.session_state.dictionary_A is None:
        st.warning("Please build the dictionary from the sidebar before uploading a test image.")
        st.stop()
        
    uploaded_file = st.file_uploader("Upload CASIA Iris Image", type=["jpg", "png", "bmp", "tiff"])
    
    if uploaded_file is not None:
        col1, col2, col3 = st.columns(3)
        
        # 1. Load and display raw image
        raw_image = process_uploaded_image(uploaded_file)
        with col1:
            st.subheader("Original Image")
            st.image(raw_image, cmap="gray", use_column_width=True)
            
        with st.spinner("Running Compressed Sensing Pipeline..."):
            try:
                # 2. Pre-process
                normalized_iris = preprocessor.process(raw_image)
                with col2:
                    st.subheader("Normalized Iris")
                    st.image(normalized_iris, cmap="gray", use_column_width=True)
                    
                # 3. Extract Features
                test_vector = extractor.extract(normalized_iris)
                
                # 4. Basis Pursuit Optimization
                sparse_x = solver.solve(st.session_state.dictionary_A, test_vector)
                
                # 5. Matching via Residuals
                matched_identity = matcher.match(
                    st.session_state.dictionary_A, 
                    test_vector, 
                    sparse_x, 
                    st.session_state.labels_array
                )
                
                with col3:
                    st.subheader("Identification Result")
                    st.success(f"**Matched Identity:**\n\n{matched_identity}")
                    
                    # Display sparsity metric for debugging/insight
                    sparsity_ratio = np.count_nonzero(np.abs(sparse_x) > 1e-5) / len(sparse_x)
                    st.caption(f"Sparsity Ratio: {sparsity_ratio:.2%}")
                    
            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")

if __name__ == "__main__":
    main()