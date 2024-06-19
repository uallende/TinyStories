import torch
import os
import sys; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import List 
from Classes.tokenizer import Tokenizer as T
import concurrent.futures
import mmap
import gzip
import numpy as np

tokenizer = T()
train_file_name = 'data/TinyStoriesV2-GPT4-train.txt'
valid_file_name = 'data/TinyStoriesV2-GPT4-valid.txt'

project_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

train_file_path = os.path.join(project_folder, train_file_name)
valid_file_path = os.path.join(project_folder, valid_file_name)
output_train_dir = 'data/tokenized_inputs/'

# Create output directory for training set if it doesn't exist
if not os.path.exists(output_train_dir):
    os.makedirs(output_train_dir)

# Parameters for chunking
chunk_size = 1024 * 1024 * 20 * 5  # 10MB, adjust as needed
buffer_size = 1024  # To ensure we don't cut words

def tokenize_and_save(text: str, output_path: str):
    tokens: List[int] = tokenizer.encode(text, bos=False, eos=False)
    token_array = np.array(tokens, dtype=np.int16)  # Use numpy.int16 instead of torch.long
    torch.save(torch.from_numpy(token_array), output_path)

def process_chunk(chunk: str, output_path: str):
    tokenize_and_save(chunk, output_path)

# Tokenize and save validation set
with open(valid_file_path, 'r', encoding='utf-8') as f:
    valid_text = f.read()
    tokenize_and_save(valid_text, f'{output_train_dir}/val.pt')

# Tokenize and save training set
with open(train_file_path, 'r', encoding='utf-8') as f:
    i = 0
    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
        while True:
            # Read chunk
            chunk = m.read(chunk_size)
            if not chunk:
                break

            # Read buffer to make sure we get the full last word in chunk
            buffer = m.read(buffer_size)
            space_pos = buffer.find(b' ')
            if space_pos != -1:
                chunk += buffer[:space_pos]
                m.seek(m.tell() - len(buffer) + space_pos + 1)  # Move cursor back

            # Process chunk in parallel
            output_path = f'{output_train_dir}/tns_chunk_{i}.pt'
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                executor.submit(process_chunk, chunk.decode('utf-8'), output_path)
            i += 1