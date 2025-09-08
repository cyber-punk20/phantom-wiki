
import argparse
import json
import os
import numpy as np


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calculate sufficient context rate")
    parser.add_argument(
        "--input_dir",
        "-id",
        help="Path to the input directory containing jsonl files",
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

def calculate_sufficient_context_rate(
        input_dir: str,
        sizes: list[int], seeds: list[int]):
    results = {}
    for size in sizes:
        rates_for_size = []
        for seed in seeds:
            sufficient_count = 0
            total_count = 0
            input_file = os.path.join(input_dir, f"depth_20_size_{size}_seed_{seed}.jsonl")
            if not os.path.exists(input_file):
                print(f"Warning: File not found, skipping: {input_file}")
                continue

            with open(input_file, 'r') as infile:
                for line in infile:
                    data = json.loads(line)
                    total_count += 1
                    statuses = data.get("statuses")
                    if statuses and statuses[-1].get('is_sufficient'):
                        sufficient_count += 1
            
            if total_count > 0:
                rates_for_size.append(sufficient_count / total_count)

        if rates_for_size:
            mean_rate = np.mean(rates_for_size)
            std_dev = np.std(rates_for_size)
            results[size] = {"mean": mean_rate, "std": std_dev}
            print(f"Size: {size}, Mean Sufficient Context Rate: {mean_rate:.4f}, Std Dev: {std_dev:.4f}")
    return results

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    results = calculate_sufficient_context_rate(args.input_dir, args.sizes, args.seeds)
    print("\nFinal results:")
    print(json.dumps(results, indent=4))