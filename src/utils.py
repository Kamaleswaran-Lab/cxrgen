import os 
from pathlib import Path


def acc_to_path(root, year, image_dir, accession_number, series_number):
    return os.path.join(root, str(year), image_dir,  f"{accession_number}_{series_number}.png")