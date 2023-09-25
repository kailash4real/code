import os
import kaggle
import pandas as pd

# Paths and URLs
DATA_DIR = "./data/info"
OUTPUT_FILE = os.path.join(DATA_DIR, "cord19_abstracts.csv")

def download_cord19_data_from_kaggle():
    """
    Download the CORD-19 dataset from Kaggle.
    Extract 20% of the abstracts and save to a CSV file.
    """
    print("Downloading CORD-19 dataset from Kaggle...")

    # Download the dataset
    kaggle.api.dataset_download_files('allen-institute-for-ai/CORD-19-research-challenge', path=DATA_DIR, unzip=True)
    
    # Assuming metadata.csv is the main file we want
    df = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))
    
    # Drop rows where abstract is NaN
    df = df.dropna(subset=['abstract'])
    
    # Extract 20% of the data
    df_sample = df.sample(frac=0.2)
    
    # Save to CSV
    df_sample['abstract'].to_csv(OUTPUT_FILE, index=False)
    print(f"Abstracts saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    download_cord19_data_from_kaggle()
