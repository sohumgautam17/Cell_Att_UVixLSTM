import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import glob
import os
import re 


def load_paired(image_path, text_path):
    # Get all files in the directory and sort them numerically
    image_files = glob.glob(os.path.join(image_path, '*'))
    text_files = glob.glob(os.path.join(text_path, '*'))


    return (sorted(image_files, key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group(0))),
            sorted(text_files, key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group(0))))


def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")


def save_result(directory_path, image_name, labels, response_text):
    # Ensure directory exists for saving analysis
    ensure_directory_exists(directory_path)

    # Save the response to a text file
    result_filename = os.path.join(directory_path, f"{image_name}_analysis.txt")
    with open(result_filename, "w") as file:
        file.write(f"Labels: {labels}\n\n")
        file.write(f'Image path: {image_name}')
        file.write(response_text)
    print(f"Analysis saved: {result_filename}")
