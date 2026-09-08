from pathlib import Path

from datasets import load_dataset


def main() -> None:
    output_dir = Path("data/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading MoDeTrans dataset...")
    dataset = load_dataset("historyHulk/MoDeTrans")

    example = dataset["train"][0]

    image = example["image"]
    ground_truth = example["text"]

    image_path = output_dir / "sample_001.png"
    text_path = output_dir / "sample_001.txt"

    image.save(image_path)
    text_path.write_text(ground_truth, encoding="utf-8")

    print()
    print("Sample extracted successfully!")
    print(f"Image: {image_path}")
    print(f"Ground truth: {text_path}")
    print()
    print("Ground truth transcription:")
    print(ground_truth)


if __name__ == "__main__":
    main()