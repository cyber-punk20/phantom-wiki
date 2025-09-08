import argparse
import json
import os
import glob

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate sca context corpus")
    parser.add_argument(
        "--output_dir", "-od", help="Path to read/write the outputs"
    )
    parser.add_argument(
        "--input_dir",
        "-id",
        help="Path to the input directory containing JSON files",
    )
    parser.add_argument(
        "--model",
        help="Model used to run sca",
    )
    parser.add_argument(
        "--sizes",
        default=[50, 500, 5000],
        metavar='N', 
        type=int, 
        nargs='+',
        help="List of dataset sizes",
    )
    parser.add_argument(
        "--seeds",
        default=[1, 2, 3],
        metavar='N', 
        type=int, 
        nargs='+',
        help="List of dataset seeds",
    )
    return parser

def generate_sca_context_corpus(
        input_dir: str, 
        output_dir: str, 
        model: str, 
        sizes: list[int], seeds: list[int]):
    os.makedirs(output_dir, exist_ok=True)
    for size in sizes:
        for seed in seeds:
            output_data = []
            pattern = f"split=depth_20_size_{size}_seed_{seed}__model_name={model}__bs=*__bn=*__seed=*.json"
            files = glob.glob(os.path.join(input_dir, pattern))
            for file in files:
                with open(file, 'r') as f:
                    data = json.load(f)
                    for key, value in data.items():
                        output_data.append({"id": key, "context": value["context"], "statuses": value["interaction"]["statuses"]})

            output_file = os.path.join(output_dir, f"depth_20_size_{size}_seed_{seed}.jsonl")
            with open(output_file, 'w') as outfile:
                for item in output_data:
                    json.dump(item, outfile)
                    outfile.write('\n')
    

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    assert(
        args.input_dir is not None
    ), "input_dir must be specified"
    assert(
        args.output_dir is not None
    ), "output_dir must be specified"
    assert(
        args.model is not None
    ), "model must be specified"
    generate_sca_context_corpus(
        args.input_dir, 
        args.output_dir, 
        args.model, 
        args.sizes, 
        args.seeds
    )