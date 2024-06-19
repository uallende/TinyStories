import torch
import os
import sys; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import List 
from Classes.tokenizer import Tokenizer as T

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

# exit(0)

# Parameters for chunking
chunk_size = 1024 * 1024 * 20 * 10 # 10MB, adjust as needed
buffer_size = 1024  # To ensure we don't cut words

def tokenize_and_save(text: str, output_path: str):
    tokens: List[int] = tokenizer.encode(text, bos=False, eos=False)
    token_tensor = torch.tensor(tokens)
    torch.save(token_tensor, output_path)

# Tokenize and save validation set
with open(valid_file_path, 'r', encoding='utf-8') as f:
    valid_text = f.read()
    tokenize_and_save(valid_text, f'{output_train_dir}/val.pt')
    
# Tokenize and save training set
with open(train_file_path, 'r', encoding='utf-8') as f:
    i = 0
    while True:
        # Read chunk
        text_chunk = f.read(chunk_size)
        if not text_chunk:
            break

        # Read buffer to make sure we get the full last word in chunk
        buffer = f.read(buffer_size)
        space_pos = buffer.find(' ')
        if space_pos != -1:
            text_chunk += buffer[:space_pos]
            f.seek(f.tell() - len(buffer) + space_pos + 1)  # Move cursor back

        # Tokenize and save
        tokenize_and_save(text_chunk, f'{output_train_dir}/tns_chunk_{i}.pt')
        i += 1
