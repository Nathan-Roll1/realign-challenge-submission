#!/usr/bin/env python3
"""
Script to download ImageNet and ObjectNet test/validation set images.
This script downloads images from:
- ImageNet test set (ILSVRC 2012)
- ImageNet validation set (ILSVRC 2012)
- ObjectNet test set
Images are stored in separate folders in the parent directory:
- imagenet/      (test set)
- imagenet_val/  (validation set)
- objectnet/
"""

import os
import sys
import argparse
import subprocess
import urllib.request
import urllib.parse
from pathlib import Path
import zipfile
import tarfile
from tqdm import tqdm
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# Configuration
PARENT_DIR = "" ### SPECIFY YOUR PARENT DIRECTORY HERE (e.g., "/path/to/datasets")
IMAGENET_DIR = os.path.join(PARENT_DIR, "imagenet")
IMAGENET_VAL_DIR = os.path.join(PARENT_DIR, "imagenet_val")
OBJECTNET_DIR = os.path.join(PARENT_DIR, "objectnet")

# URLs
# Note: ImageNet URLs may require manual download from https://www.image-net.org/download
# The direct URLs may not work due to authentication/session requirements
IMAGENET_TEST_URL = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_test_v10102019.tar"
IMAGENET_VAL_URL = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"
OBJECTNET_URL = "https://objectnet.dev/downloads/objectnet-1.0.zip"
OBJECTNET_PASSWORD = "objectnetisatestset"


class DownloadProgressBar:
    """Progress bar for file downloads."""
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading")
        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(block_size)
        else:
            self.pbar.close()


def download_file(url, output_path, username=None, password=None):
    """Download a file from URL with optional authentication."""
    print(f"Downloading from {url}...")
    print(f"Saving to {output_path}...")
    
    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Try using requests library if available (better for authentication)
    if REQUESTS_AVAILABLE and username and password:
        try:
            session = requests.Session()
            session.auth = (username, password)
            
            response = session.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            with open(output_path, 'wb') as f, tqdm(
                desc="Downloading",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            print(f"\nDownload completed: {output_path}")
            return True
        except requests.exceptions.RequestException as e:
            print(f"\nError downloading with requests: {e}")
            # Fall back to urllib method
    
    # Fallback to urllib method
    if username and password:
        password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        password_mgr.add_password(None, url, username, password)
        handler = urllib.request.HTTPBasicAuthHandler(password_mgr)
        opener = urllib.request.build_opener(handler)
        urllib.request.install_opener(opener)
    
    try:
        urllib.request.urlretrieve(url, output_path, reporthook=DownloadProgressBar())
        print(f"\nDownload completed: {output_path}")
        return True
    except urllib.error.HTTPError as e:
        if e.code == 401:
            print(f"\nError: Authentication failed. Please check your ImageNet credentials.")
        else:
            print(f"\nError downloading file: {e}")
        return False
    except Exception as e:
        print(f"\nError downloading file: {e}")
        return False


def extract_tar(tar_path, extract_dir):
    """Extract a tar archive."""
    print(f"Extracting {tar_path} to {extract_dir}...")
    os.makedirs(extract_dir, exist_ok=True)
    
    try:
        with tarfile.open(tar_path, 'r') as tar:
            # Get total number of members for progress bar
            members = tar.getmembers()
            with tqdm(total=len(members), desc="Extracting") as pbar:
                for member in members:
                    tar.extract(member, extract_dir)
                    pbar.update(1)
        print(f"Extraction completed: {extract_dir}")
        return True
    except Exception as e:
        print(f"Error extracting tar file: {e}")
        return False


def extract_zip(zip_path, extract_dir, password=None):
    """Extract a zip archive with optional password.
    
    Uses system 'unzip' command for faster extraction (especially for 
    password-protected archives), falls back to Python zipfile if unavailable.
    """
    print(f"Extracting {zip_path} to {extract_dir}...")
    os.makedirs(extract_dir, exist_ok=True)
    
    # Try using system unzip command (much faster, especially for encrypted zips)
    try:
        cmd = ["unzip", "-o"]  # -o: overwrite without prompting
        if password:
            cmd.extend(["-P", password])
        cmd.extend([zip_path, "-d", extract_dir])
        
        print("Using system 'unzip' for faster extraction...")
        # Disable zip bomb detection (ObjectNet has overlapped components that trigger false positive)
        env = os.environ.copy()
        env["UNZIP_DISABLE_ZIPBOMB_DETECTION"] = "TRUE"
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        
        if result.returncode == 0:
            print(f"Extraction completed: {extract_dir}")
            return True
        else:
            print(f"unzip failed (exit code {result.returncode}), falling back to Python...")
            if result.stderr:
                print(f"Error: {result.stderr}")
    except FileNotFoundError:
        print("System 'unzip' not found, using Python zipfile (slower)...")
    except Exception as e:
        print(f"Error with system unzip: {e}, falling back to Python...")
    
    # Fallback to Python zipfile (slower but always available)
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            if password:
                zip_ref.setpassword(password.encode())
            
            # Get total number of files for progress bar
            file_list = zip_ref.namelist()
            with tqdm(total=len(file_list), desc="Extracting") as pbar:
                for file in file_list:
                    zip_ref.extract(file, extract_dir)
                    pbar.update(1)
        print(f"Extraction completed: {extract_dir}")
        return True
    except zipfile.BadZipFile as e:
        print(f"Error: Invalid zip file or wrong password: {e}")
        return False
    except Exception as e:
        print(f"Error extracting zip file: {e}")
        return False


def is_imagenet_extracted(imagenet_dir):
    """Check if ImageNet test set is already extracted."""
    if not os.path.exists(imagenet_dir):
        return False
    
    # Look for extracted image files (JPEG files)
    # ImageNet test set typically has files like ILSVRC2012_test_*.JPEG
    for root, dirs, files in os.walk(imagenet_dir):
        # Skip the tar file itself
        if any(f.endswith('.tar') for f in files):
            continue
        # Check for image files
        image_files = [f for f in files if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
        if image_files:
            return True
    
    return False


def download_imagenet(username=None, password=None, skip_download=False, non_interactive=False):
    """Download and extract ImageNet test set."""
    print("\n" + "="*60)
    print("Downloading ImageNet Test Set")
    print("="*60)
    
    tar_path = os.path.join(IMAGENET_DIR, "ILSVRC2012_img_test.tar")
    
    # Check if already extracted
    if is_imagenet_extracted(IMAGENET_DIR):
        print(f"✓ ImageNet test set is already extracted in: {IMAGENET_DIR}")
        print("Skipping download and extraction.")
        return True
    
    # Check if tar file exists but not extracted
    if os.path.exists(tar_path):
        print(f"Found existing tar file: {tar_path}")
        if not skip_download:
            if not non_interactive:
                try:
                    response = input("Extract existing tar file? (y/n): ").strip().lower()
                    if response != 'y':
                        return False
                except (EOFError, KeyboardInterrupt):
                    print("Non-interactive mode: automatically extracting...")
            # Extract the tar file
            success = extract_tar(tar_path, IMAGENET_DIR)
            if success and not non_interactive:
                # Optionally remove tar file to save space
                print(f"\nTar file saved at: {tar_path}")
                try:
                    response = input("Remove tar file to save space? (y/n): ").strip().lower()
                    if response == 'y':
                        os.remove(tar_path)
                        print("Tar file removed.")
                except (EOFError, KeyboardInterrupt):
                    print("Keeping tar file (non-interactive mode)")
            return success
    
    # Download if needed
    if not skip_download and not os.path.exists(tar_path):
        if not username or not password:
            print("\nImageNet requires authentication.")
            print("Please provide your ImageNet username and password.")
            print("You can set them via environment variables:")
            print("  export IMAGENET_USERNAME='your_username'")
            print("  export IMAGENET_PASSWORD='your_password'")
            print("\nOr provide them via command-line arguments:")
            print("  --imagenet-username USERNAME --imagenet-password PASSWORD")
            print("\nAlternatively, you can manually download from:")
            print("  https://www.image-net.org/download")
            print(f"\nAnd place ILSVRC2012_img_test.tar in: {IMAGENET_DIR}")
            return False
        
        if not download_file(IMAGENET_TEST_URL, tar_path, username, password):
            print("\n" + "="*60)
            print("ImageNet download failed. This is common because:")
            print("1. The direct URL may require a session-based download link")
            print("2. You may need to download manually from the ImageNet website")
            print("\nTo download manually:")
            print("1. Visit: https://www.image-net.org/download")
            print("2. Log in with your credentials")
            print("3. Download 'ILSVRC2012_img_test.tar'")
            print(f"4. Place it in: {IMAGENET_DIR}")
            print("5. Run this script again with --skip-download to extract it")
            print("="*60)
            return False
    
    # Extract if tar file exists
    if os.path.exists(tar_path):
        success = extract_tar(tar_path, IMAGENET_DIR)
        if success and not non_interactive:
            # Optionally remove tar file to save space
            print(f"\nTar file saved at: {tar_path}")
            try:
                response = input("Remove tar file to save space? (y/n): ").strip().lower()
                if response == 'y':
                    os.remove(tar_path)
                    print("Tar file removed.")
            except (EOFError, KeyboardInterrupt):
                print("Keeping tar file (non-interactive mode)")
        return success
    else:
        print(f"Error: Tar file not found at {tar_path}")
        return False


def is_objectnet_extracted(objectnet_dir):
    """Check if ObjectNet test set is already extracted."""
    if not os.path.exists(objectnet_dir):
        return False
    
    # Look for extracted files/directories (not just the zip file)
    all_items = os.listdir(objectnet_dir)
    # Filter out the zip file itself
    extracted_items = [item for item in all_items if item != "objectnet-1.0.zip"]
    
    if not extracted_items:
        return False
    
    # Check if there are actual image files or directories with images
    for item in extracted_items:
        item_path = os.path.join(objectnet_dir, item)
        if os.path.isdir(item_path):
            # Check if directory contains image files
            for root, dirs, files in os.walk(item_path):
                image_files = [f for f in files if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
                if image_files:
                    return True
        elif os.path.isfile(item_path):
            # Check if it's an image file
            if item.lower().endswith(('.jpeg', '.jpg', '.png')):
                return True
    
    return False


def is_imagenet_val_extracted(imagenet_val_dir):
    """Check if ImageNet validation set is already extracted."""
    if not os.path.exists(imagenet_val_dir):
        return False
    
    # Look for extracted image files (JPEG files)
    # ImageNet validation set typically has files like ILSVRC2012_val_*.JPEG
    for root, dirs, files in os.walk(imagenet_val_dir):
        # Skip the tar file itself
        if any(f.endswith('.tar') for f in files):
            continue
        # Check for image files
        image_files = [f for f in files if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
        if image_files:
            return True
    
    return False


def download_imagenet_val(username=None, password=None, skip_download=False, non_interactive=False):
    """Download and extract ImageNet validation set."""
    print("\n" + "="*60)
    print("Downloading ImageNet Validation Set")
    print("="*60)
    
    tar_path = os.path.join(IMAGENET_VAL_DIR, "ILSVRC2012_img_val.tar")
    
    # Check if already extracted
    if is_imagenet_val_extracted(IMAGENET_VAL_DIR):
        print(f"✓ ImageNet validation set is already extracted in: {IMAGENET_VAL_DIR}")
        print("Skipping download and extraction.")
        return True
    
    # Check if tar file exists but not extracted
    if os.path.exists(tar_path):
        print(f"Found existing tar file: {tar_path}")
        if not skip_download:
            if not non_interactive:
                try:
                    response = input("Extract existing tar file? (y/n): ").strip().lower()
                    if response != 'y':
                        return False
                except (EOFError, KeyboardInterrupt):
                    print("Non-interactive mode: automatically extracting...")
            # Extract the tar file
            success = extract_tar(tar_path, IMAGENET_VAL_DIR)
            if success and not non_interactive:
                # Optionally remove tar file to save space
                print(f"\nTar file saved at: {tar_path}")
                try:
                    response = input("Remove tar file to save space? (y/n): ").strip().lower()
                    if response == 'y':
                        os.remove(tar_path)
                        print("Tar file removed.")
                except (EOFError, KeyboardInterrupt):
                    print("Keeping tar file (non-interactive mode)")
            return success
    
    # Download if needed
    if not skip_download and not os.path.exists(tar_path):
        if not username or not password:
            print("\nImageNet requires authentication.")
            print("Please provide your ImageNet username and password.")
            print("You can set them via environment variables:")
            print("  export IMAGENET_USERNAME='your_username'")
            print("  export IMAGENET_PASSWORD='your_password'")
            print("\nOr provide them via command-line arguments:")
            print("  --imagenet-username USERNAME --imagenet-password PASSWORD")
            print("\nAlternatively, you can manually download from:")
            print("  https://www.image-net.org/download")
            print(f"\nAnd place ILSVRC2012_img_val.tar in: {IMAGENET_VAL_DIR}")
            return False
        
        if not download_file(IMAGENET_VAL_URL, tar_path, username, password):
            print("\n" + "="*60)
            print("ImageNet validation download failed. This is common because:")
            print("1. The direct URL may require a session-based download link")
            print("2. You may need to download manually from the ImageNet website")
            print("\nTo download manually:")
            print("1. Visit: https://www.image-net.org/download")
            print("2. Log in with your credentials")
            print("3. Download 'ILSVRC2012_img_val.tar'")
            print(f"4. Place it in: {IMAGENET_VAL_DIR}")
            print("5. Run this script again with --skip-download to extract it")
            print("="*60)
            return False
    
    # Extract if tar file exists
    if os.path.exists(tar_path):
        success = extract_tar(tar_path, IMAGENET_VAL_DIR)
        if success and not non_interactive:
            # Optionally remove tar file to save space
            print(f"\nTar file saved at: {tar_path}")
            try:
                response = input("Remove tar file to save space? (y/n): ").strip().lower()
                if response == 'y':
                    os.remove(tar_path)
                    print("Tar file removed.")
            except (EOFError, KeyboardInterrupt):
                print("Keeping tar file (non-interactive mode)")
        return success
    else:
        print(f"Error: Tar file not found at {tar_path}")
        return False


def download_objectnet(skip_download=False, non_interactive=False):
    """Download and extract ObjectNet test set."""
    print("\n" + "="*60)
    print("Downloading ObjectNet Test Set")
    print("="*60)
    
    zip_path = os.path.join(OBJECTNET_DIR, "objectnet-1.0.zip")
    
    # Check if already extracted
    if is_objectnet_extracted(OBJECTNET_DIR):
        print(f"✓ ObjectNet test set is already extracted in: {OBJECTNET_DIR}")
        print("Skipping download and extraction.")
        return True
    
    # Check if zip file exists but not extracted
    if os.path.exists(zip_path):
        print(f"Found existing zip file: {zip_path}")
        if not skip_download:
            if not non_interactive:
                try:
                    response = input("Extract existing zip file? (y/n): ").strip().lower()
                    if response != 'y':
                        return False
                except (EOFError, KeyboardInterrupt):
                    print("Non-interactive mode: automatically extracting...")
            # Extract the zip file
            success = extract_zip(zip_path, OBJECTNET_DIR, OBJECTNET_PASSWORD)
            if success and not non_interactive:
                # Optionally remove zip file to save space
                print(f"\nZip file saved at: {zip_path}")
                try:
                    response = input("Remove zip file to save space? (y/n): ").strip().lower()
                    if response == 'y':
                        os.remove(zip_path)
                        print("Zip file removed.")
                except (EOFError, KeyboardInterrupt):
                    print("Keeping zip file (non-interactive mode)")
            return success
    
    # Download if needed
    if not skip_download and not os.path.exists(zip_path):
        if not download_file(OBJECTNET_URL, zip_path):
            return False
    
    # Extract if zip file exists
    if os.path.exists(zip_path):
        success = extract_zip(zip_path, OBJECTNET_DIR, OBJECTNET_PASSWORD)
        if success and not non_interactive:
            # Optionally remove zip file to save space
            print(f"\nZip file saved at: {zip_path}")
            try:
                response = input("Remove zip file to save space? (y/n): ").strip().lower()
                if response == 'y':
                    os.remove(zip_path)
                    print("Zip file removed.")
            except (EOFError, KeyboardInterrupt):
                print("Keeping zip file (non-interactive mode)")
        return success
    else:
        print(f"Error: Zip file not found at {zip_path}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download ImageNet and ObjectNet test set images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all datasets (ImageNet requires credentials)
  python download_test_sets.py --imagenet-username USER --imagenet-password PASS
  # Download only ObjectNet
  python download_test_sets.py --objectnet-only
  # Download only ImageNet test set
  python download_test_sets.py --imagenet-only --imagenet-username USER --imagenet-password PASS
  # Download only ImageNet validation set
  python download_test_sets.py --imagenet-val-only --imagenet-username USER --imagenet-password PASS
  # Download all but skip ImageNet validation set
  python download_test_sets.py --skip-imagenet-val --imagenet-username USER --imagenet-password PASS
  # Use environment variables for ImageNet credentials
  export IMAGENET_USERNAME='your_username'
  export IMAGENET_PASSWORD='your_password'
  python download_test_sets.py
        """
    )
    
    parser.add_argument(
        '--imagenet-username',
        type=str,
        default=os.environ.get('IMAGENET_USERNAME'),
        help='ImageNet username (or set IMAGENET_USERNAME env var)'
    )
    parser.add_argument(
        '--imagenet-password',
        type=str,
        default=os.environ.get('IMAGENET_PASSWORD'),
        help='ImageNet password (or set IMAGENET_PASSWORD env var)'
    )
    parser.add_argument(
        '--imagenet-only',
        action='store_true',
        help='Download only ImageNet test set'
    )
    parser.add_argument(
        '--imagenet-val-only',
        action='store_true',
        help='Download only ImageNet validation set'
    )
    parser.add_argument(
        '--objectnet-only',
        action='store_true',
        help='Download only ObjectNet test set'
    )
    parser.add_argument(
        '--skip-imagenet-val',
        action='store_true',
        help='Skip ImageNet validation set download'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip download, only extract existing archives'
    )
    parser.add_argument(
        '--non-interactive',
        action='store_true',
        help='Run in non-interactive mode (skip all prompts, auto-extract)'
    )
    parser.add_argument(
        '--parent-dir',
        type=str,
        default=PARENT_DIR,
        help=f'Parent directory for storing datasets (default: {PARENT_DIR})'
    )
    
    args = parser.parse_args()
    
    # Update directories if custom parent dir is provided
    global IMAGENET_DIR, IMAGENET_VAL_DIR, OBJECTNET_DIR
    if args.parent_dir != PARENT_DIR:
        IMAGENET_DIR = os.path.join(args.parent_dir, "imagenet")
        IMAGENET_VAL_DIR = os.path.join(args.parent_dir, "imagenet_val")
        OBJECTNET_DIR = os.path.join(args.parent_dir, "objectnet")
    
    print(f"Parent directory: {args.parent_dir}")
    print(f"ImageNet test directory: {IMAGENET_DIR}")
    print(f"ImageNet val directory: {IMAGENET_VAL_DIR}")
    print(f"ObjectNet directory: {OBJECTNET_DIR}")
    
    results = {}
    
    # Determine which datasets to download based on flags
    download_imagenet_test = not args.objectnet_only and not args.imagenet_val_only
    download_imagenet_validation = not args.objectnet_only and not args.imagenet_only and not args.skip_imagenet_val
    download_objectnet_set = not args.imagenet_only and not args.imagenet_val_only
    
    # Handle --imagenet-val-only flag
    if args.imagenet_val_only:
        download_imagenet_test = False
        download_imagenet_validation = True
        download_objectnet_set = False
    
    # Download ImageNet test set
    if download_imagenet_test:
        results['imagenet'] = download_imagenet(
            username=args.imagenet_username,
            password=args.imagenet_password,
            skip_download=args.skip_download,
            non_interactive=args.non_interactive
        )
    
    # Download ImageNet validation set
    if download_imagenet_validation:
        results['imagenet_val'] = download_imagenet_val(
            username=args.imagenet_username,
            password=args.imagenet_password,
            skip_download=args.skip_download,
            non_interactive=args.non_interactive
        )
    
    # Download ObjectNet
    if download_objectnet_set:
        results['objectnet'] = download_objectnet(skip_download=args.skip_download, non_interactive=args.non_interactive)
    
    # Summary
    print("\n" + "="*60)
    print("Download Summary")
    print("="*60)
    for dataset, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"{dataset.capitalize()}: {status}")
    
    if all(results.values()):
        print("\nAll datasets downloaded successfully!")
        return 0
    else:
        print("\nSome downloads failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        sys.exit(1)
