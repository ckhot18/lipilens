from models.qwen_modi import ModiTranscriber


IMAGE_PATH = "data/evaluation/sample_001.png"


def main() -> None:
    transcriber = ModiTranscriber()

    result = transcriber.transcribe(IMAGE_PATH)

    print("\n" + "=" * 60)
    print("AI TRANSCRIPTION")
    print("=" * 60)
    print(result)
    print("=" * 60)


if __name__ == "__main__":
    main()