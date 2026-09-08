from datasets import load_dataset


def main() -> None:
    dataset = load_dataset("historyHulk/MoDeTrans")

    print(dataset)
    print(f"Number of examples: {len(dataset['train'])}")


if __name__ == "__main__":
    main()