import sys
import os.path
import argparse

import torch

from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer


def contains_non_latin1_characters(s):
    try:
        # If this operation fails, it means there was a problem encoding the string to UTF-8,
        # which should not happen with properly formed Unicode strings in Python.
        s.encode('latin-1')
    except UnicodeEncodeError:
        return True
    # No exception means all characters can be encoded in UTF-8.
    return False


def get_batch_from_raw(data, batch_size=32, seq_len=4):
    # To use if dataset is a whole list of tokens
    # gets a random idx
    idx = torch.randint(0, len(data) - seq_len - 1, (batch_size,))

    x = torch.stack([data[i:i+seq_len] for i in idx])
    y = torch.stack([data[i+1:i+seq_len+1] for i in idx])
    return x, y


def get_seq_batch(data, batch_size=32, seq_len=4):
    # To use if dataset is a whole list of tokens
    # gets a random idx
    idx = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i][:seq_len] for i in idx])
    y = torch.stack([data[i][1:seq_len+1] for i in idx])
    return x, y


def process_dataset(dataset, dataset_path, dataset_len=10000, seq_len=4,
                    trained_tokenizer_path="",  truncation=True, padding=True, 
                    tokenizer_params={}):

    # Transfer streaming dataset to local file .txt
    if not os.path.exists(dataset_path):
        with open(dataset_path, "w", encoding="utf-8") as f:
            for i,s in enumerate(dataset):
                if (len(s["text"]) > 20 ) and not contains_non_latin1_characters(s["text"]):
                    f.write(s["text"])
                if i > dataset_len:
                    break

    # Initialize a tokenizer
    if not trained_tokenizer_path:
        tokenizer = ByteLevelBPETokenizer()

        # Train the tokenizer
        print("Training tokenizer")
        tokenizer.train(files=dataset_path, special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ], **tokenizer_params)

        # Save the trained tokenizer
        tokenizer.save_model(".", "bytelevel_bpe")
    else:
        tokenizer = ByteLevelBPETokenizer(
            trained_tokenizer_path + "-vocab.json",
            trained_tokenizer_path + "-merges.txt",
        )

    # truncation and padding
    if truncation:
        tokenizer.enable_truncation(max_length=seq_len+1)
    if padding:
        tokenizer.enable_padding(length=seq_len+1)

    print("Encoding the dataset")
    with open(dataset_path, "r") as f:
        dataset = [tokenizer.encode(string).ids for string in f]

    # dataset to tensor
    dataset = torch.tensor(dataset, dtype=torch.long)

    print("Done")
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Process a dataset.")
    parser.add_argument('--dataset', type=None, required=True, help='The dataset (HF streaming dataset)')
    parser.add_argument('--dataset_path', type=str, required=False, help='Path to save the dataset (.txt)')
    parser.add_argument('--trained_tokenizer', type=None, required=False, help='The trained tokenizer path')
    parser.add_argument('--truncation', type=bool, required=False, help='Enable truncation')
    parser.add_argument('--padding', type=bool, required=False, help='Enable padding')
    parser.add_argument('seq_len', type=int, required=False, help='The sequence length, default 4')
    parser.add_argument('dataset_len', type=int, required=False, help='The dataset length, default 10000')
    parser.add_argument('tokenizer_params', type=dict, required=False, help='The tokenizer parameters')

    args = parser.parse_args()

    # Process the dataset
    result = process_dataset(args.dataset, args.dataset_path,
                             args.dataset_len, 
                             args.seq_len, args.trained_tokenizer, 
                             args.truncation, args.padding, 
                             args.tokenizer_params)


if main == "__main__":
    main()
    

