from tfmbench.dataset.OpenML import *
import argparse

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--folder",
        required=True,
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
    )

    args = parser.parse_args()
    ds = OpenMLDataset(args.data)
    data = ds.load()
    ds.save(args.folder)

if __name__ == "__main__":
    main()