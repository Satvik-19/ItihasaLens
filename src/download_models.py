"""
Script to download model files from Google Drive.
"""

import os
import gdown
from tqdm import tqdm

# Model file IDs from Google Drive
MODEL_FILES = {
    'monument_recognition_model.h5': '1oXy582p8m0OrWRIS8xTXB4MyKmS5iLhx',
    'damage_detection_model.h5': '1lmyGT_Fn2Hqs1SaPoKcSLrfYmDRhbTYt',
}

def download_file_from_google_drive(file_id, destination):
    """Download a file from Google Drive."""
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, destination, quiet=False)

def download_models():
    """Download all model files."""
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    print("Downloading model files...")
    for model_name, file_id in MODEL_FILES.items():
        destination = os.path.join('models', model_name)
        print(f"\nDownloading {model_name}...")
        try:
            download_file_from_google_drive(file_id, destination)
            print(f"Successfully downloaded {model_name}")
        except Exception as e:
            print(f"Error downloading {model_name}: {str(e)}")
            print("Please download manually from the provided link")

def main():
    """Main function to download models."""
    print("Starting model download process...")
    download_models()
    print("\nDownload complete!")
    print("\nNote: If automatic download fails, please download the models manually from:")
    print("Google Drive Link: https://drive.google.com/drive/folders/1qjKKhYTsAtPscq_lEcBk3HG1GcENw1e_?usp=sharing")
    print("\nPlace the downloaded models in the 'models/' directory.")

if __name__ == "__main__":
    main() 