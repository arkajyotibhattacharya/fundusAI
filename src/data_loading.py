"""
Data loading: Google Drive mount or Kaggle download, path resolution.
"""

import os
import glob
import shutil
import pandas as pd


def mount_drive():
    """Mount Google Drive in Colab and return the base path."""
    from google.colab import drive
    drive.mount('/content/drive')
    return '/content/drive/MyDrive'


def load_odir(path='ocular_data'):
    """
    Load ODIR-5K from a local path.
    Works with any location:
      - Colab runtime upload: load_odir('ocular_data')
      - Google Drive:         load_odir('/content/drive/MyDrive/FundusAI/ocular_data')
    """
    data_ocu = pd.read_csv(os.path.join(path, 'full_df.csv'))

    image_dir = os.path.join(path, 'ODIR-5K')
    data_ocu['paths'] = data_ocu['filepath'].apply(
        lambda x: os.path.normpath(os.path.join(image_dir, '/'.join(x.split('/')[3:])))
    )
    print(f"Loaded {len(data_ocu)} records from {path}")
    return data_ocu


def authenticate_kaggle(username, api_key):
    """Set Kaggle credentials and return an authenticated API client."""
    os.environ['KAGGLE_USERNAME'] = username
    os.environ['KAGGLE_KEY'] = api_key

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    print("Authentication successful!")
    return api


def download_odir(api, download_path='./ocular_data'):
    """Download ODIR-5K via Kaggle and return a DataFrame with corrected image paths."""
    dataset_identifier = 'andrewmvd/ocular-disease-recognition-odir5k'
    api.dataset_download_files(dataset_identifier, path=download_path, unzip=True)
    print("Download complete!")

    data_ocu = pd.read_csv(os.path.join(download_path, 'full_df.csv'))

    image_dir = os.path.join(download_path, 'ODIR-5K')
    data_ocu['paths'] = data_ocu['filepath'].apply(
        lambda x: os.path.normpath(os.path.join(image_dir, '/'.join(x.split('/')[3:])))
    )
    return data_ocu


def download_cataract(api, download_path='./cataract_data'):
    """Download and prepare the cataract dataset (optional augmentation)."""
    dataset_identifier = 'jr2ngb/cataractdataset'
    api.dataset_download_files(dataset_identifier, path=download_path, unzip=True)

    # Clean up nested folder structure
    base_path = download_path
    inner_folder = os.path.join(base_path, 'dataset')
    temp_path = './temp_storage'

    shutil.move(inner_folder, temp_path)
    shutil.rmtree(base_path)
    os.rename(temp_path, base_path)
    print("Download complete! Check the 'cataract_data' folder on the left sidebar.")

    # Build DataFrame
    filepaths = []
    labels = []

    label_map = {
        '1_normal': '0',
        '2_cataract': '1',
        '2_glaucoma': '2',
        '3_retina_disease': '3',
    }

    for folder_name, label_value in label_map.items():
        folder_path = os.path.join(download_path, folder_name)
        for img_path in glob.glob(os.path.join(folder_path, '*')):
            filepaths.append(img_path)
            labels.append(label_value)

    cat_df = pd.DataFrame({'paths': filepaths, 'cataract': labels})

    # Only keep normal and cataract
    cat_df = cat_df[(cat_df['cataract'] == '0') | (cat_df['cataract'] == '1')]
    return cat_df
