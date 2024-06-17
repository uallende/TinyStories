import os
import subprocess
import sys

# Find the root folder of the Git repository
try:
    root_folder = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode('utf-8').strip()
except subprocess.CalledProcessError:
    print("Error: Not a git repository.")
    sys.exit(1)

# Set the size limit (in bytes). Here, 1MB = 1 * 1024 * 1024 bytes
size_limit = 10 * 1024 * 1024
large_files = []

for foldername, subfolders, filenames in os.walk(root_folder):
    for filename in filenames:
        filepath = os.path.join(foldername, filename)
        
        if '.git' in filepath:
            continue
        file_size = os.path.getsize(filepath)
        
        if file_size > size_limit:
            large_files.append(filepath)

# Append to .gitignore in the root folder
with open(os.path.join(root_folder, '.gitignore'), 'a') as f:
    for large_file in large_files:
        f.write(f"{large_file}\n")

print(f"Added the following large files to .gitignore:\n{large_files}")

