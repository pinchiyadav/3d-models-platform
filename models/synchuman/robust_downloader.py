import os
import sys
import logging
from huggingface_hub import hf_hub_download, list_repo_files
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

REPO_ID = "xishushu/SyncHuman"

def download_file_with_retry(repo_id, filepath, local_dir, token, retries=5):
    """Downloads a single file with retries."""
    for attempt in range(retries):
        logging.info(f"Downloading {filepath} (attempt {attempt + 1}/{retries})...")
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filepath,
                repo_type='model',
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                token=token,
                resume_download=True,
                etag_timeout=60, # timeout for checking ETag
            )
            logging.info(f"Successfully downloaded {filepath}")
            return True
        except Exception as e:
            logging.error(f"Download of {filepath} failed on attempt {attempt + 1}: {e}")
            if "401" in str(e):
                logging.error("Got 401 Unauthorized. Check your HF_TOKEN.")
                break # don't retry on auth error
            if attempt < retries - 1:
                logging.info("Retrying in 10 seconds...")
                time.sleep(10)
            else:
                logging.error(f"Failed to download {filepath} after {retries} attempts.")
                return False
    return False

def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        logging.error("HF_TOKEN environment variable not set.")
        sys.exit(1)

    sync_root = os.environ.get("SYNC_ROOT", "/workspace/SyncHuman")
    
    try:
        files_to_download = list_repo_files(repo_id=REPO_ID, repo_type='model', token=token)
    except Exception as e:
        logging.error(f"Could not list files in repo {REPO_ID}: {e}")
        sys.exit(1)
        
    logging.info(f"Found {len(files_to_download)} files to download from {REPO_ID}.")

    # Filter out dotfiles
    files_to_download = [f for f in files_to_download if not Path(f).name.startswith('.')]

    success = True
    for filepath in files_to_download:
        if not download_file_with_retry(REPO_ID, filepath, sync_root, token):
            success = False
    
    if not success:
        logging.error("One or more files failed to download. Please check the logs.")
        sys.exit(1)
        
    logging.info("All download tasks completed successfully.")

if __name__ == "__main__":
    main()
