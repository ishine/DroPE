#!/usr/bin/env python
import argparse
from transformers import AutoModel


def main():
    parser = argparse.ArgumentParser(
        description="load and print a huggingface model on cpu"
    )
    parser.add_argument(
        "model_name_or_path",
        help="name or path of the model to load"
    )
    args = parser.parse_args()

    model_ref = args.model_name_or_path
    # try loading from local cache only
    try:
        print(f"loading {model_ref} from local cache only...")
        model = AutoModel.from_pretrained(model_ref, local_files_only=True)
        print("model loaded from local cache")
    except Exception:
        print("model not in cache; downloading...")
        model = AutoModel.from_pretrained(model_ref)
        print("model downloaded and loaded")

    print(model)


if __name__ == "__main__":
    main()
