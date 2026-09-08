from huggingface_hub import snapshot_download


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


def main() -> None:
    print(f"Downloading model: {MODEL_ID}")
    print("Files will be cached by Hugging Face.")

    path = snapshot_download(
        repo_id=MODEL_ID,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.model",
            "*.tiktoken",
            "*.txt",
            "*.py",
        ],
    )

    print()
    print("Download complete.")
    print(f"Cached at: {path}")


if __name__ == "__main__":
    main()