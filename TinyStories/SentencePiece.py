import sentencepiece as spm

# Define the training parameters
input_file = 'data/TinyStoriesV2-GPT4-train.txt'
model_prefix = 'tokenizer'
vocab_size = 15000
model_type = 'bpe'

# Train the model using a dictionary for arguments
spm.SentencePieceTrainer.train(input=input_file, model_prefix=model_prefix, vocab_size=vocab_size, model_type=model_type)
