# https://huggingface.co/gpt2/raw/main/vocab.json
import ast

vocab_file = "/home/helen/data/vocab.json"

with open(vocab_file, 'r') as f:
    vocab_dict = ast.literal_eval(f.read())

