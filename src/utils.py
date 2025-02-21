import os 
from pathlib import Path
import hashlib

def hash_value(value, hash_key = '123'):
    return hashlib.sha256((str(value) + hash_key).encode()).hexdigest()


def acc_to_path(root, year, image_dir, accession_number, series_number):
    return os.path.join(root, str(year), image_dir,  f"{accession_number}_{series_number}.png")