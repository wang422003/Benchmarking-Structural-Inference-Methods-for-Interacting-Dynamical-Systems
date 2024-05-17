# Auto-download datasets in Google Drive

This folder contains a script to download all datasets from Google Drive and unzip the files to a proper location automatically.
This folder also serves as a temporary file for storing all dataset zip files.

To download datasets selectively, please comment out unwanted datasets in both `download_dict` and `unzip_dict` in the script.

The downloaded zip files might be corrupted during transmission.
Re-running the script will detect this, remove the corrupted files, and re-download files automatically.
However, this feature does not support multiple datasets sharing the same directory, i.e., the brain network dataset, as there are additional datasets for noise trajectories.

This script serves as a helper for downloading the dataset files in batch.
If this script fails, please manually download files from the [download page of StructInfer](https://structinfer.github.io/download/).

This script was written with reference to [this page](https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url).