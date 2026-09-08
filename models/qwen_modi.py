from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)


BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
MODI_ADAPTER = "lgtk/qwen25vl-3b-modi-synth-lora"

PROMPT = (
    "This image contains handwritten text in Modi script, a historical cursive "
    "script used to write the Marathi language. "
    "Transliterate the text in this image into Devanagari script. "
    "Output only the Devanagari text, with no explanation."
)


class ModiTranscriber:
    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU was not detected. For this first test, use a CUDA-enabled "
                "environment such as your NVIDIA GPU or Google Colab."
            )

        print(f"GPU: {torch.cuda.get_device_name(0)}")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        print("Loading Qwen base model...")
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            BASE_MODEL,
            quantization_config=quantization_config,
            device_map="auto",
        )

        print("Loading Modi adapter...")
        self.model = PeftModel.from_pretrained(
            base_model,
            MODI_ADAPTER,
        )

        self.model.eval()

        # Keep visual input deliberately small for your 4 GB GPU.
        self.processor = AutoProcessor.from_pretrained(
            BASE_MODEL,
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 28 * 28,
        )

        print("Model loaded.")

    @torch.inference_mode()
    def transcribe(self, image_path: str | Path) -> str:
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": PROMPT,
                    },
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        )

        # device_map="auto" decides where the model lives.
        # Move inputs to the same device used by the model.
        inputs = {
            key: value.to(self.model.device)
            if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )

        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]

        result = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        return result