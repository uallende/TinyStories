import subprocess

def get_large_files():
    try:
        # Step 1: Get a list of all blob hashes
        blob_hashes = subprocess.check_output([
            "git", "rev-list", "--objects", "--all"
        ]).decode('utf-8').strip().split('\n')
        blob_hashes = [line.split()[0] for line in blob_hashes]

        large_files = []

        # Step 2: Get the size of each blob and check if it's large
        for blob in blob_hashes:
            size = int(subprocess.check_output([
                "git", "cat-file", "-s", blob
            ]).decode('utf-8').strip())
            
            if size > 10 * 1024 * 1024:  # 10 MB
                large_files.append(blob)

        return large_files

    except subprocess.CalledProcessError:
        print("Error: Not a git repository.")
        return []

# Get list of large blob hashes
large_files = get_large_files()

# Note: The blobs won't correspond directly to file paths, so unstaging won't work as-is.
# This is just to demonstrate how to find large blobs.
print(f"Large blobs: {large_files}")
