from typing import List

from PIL import Image
import torch
import pdb

from open_flamingo.eval.eval_model import BaseEvalModel
from contextlib import suppress

try:
    # Prefer transformers-based loading; many vendor models (including Qwen)
    # expose remote code that supports vision_x in generate when
    # `trust_remote_code=True` is used.
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
    from torchvision import transforms
except Exception:
    Qwen2_5_VLForConditionalGeneration = None
    AutoTokenizer = None
    AutoProcessor = None
    transforms = None

import re

def flamingo_prompt_to_qwen_messages(prompt: str, images):
    chunks = [c.strip() for c in prompt.split("<image>") if c.strip()]

    if len(chunks) != len(images):
        raise ValueError(
            f"Mismatch: {len(chunks)} <image> chunks vs {len(images)} images"
        )

    messages = []

    for i, (chunk, image) in enumerate(zip(chunks, images)):
        chunk = chunk.replace("<|endofchunk|>", "").strip()

        # match "Short answer", possibly with extra description
        match = re.search(r"Short answer.*?:", chunk)
        if match is None:
            raise ValueError(f"Malformed chunk (no Short answer): {chunk}")

        split_idx = match.start()
        qa_sep_end = match.end()

        q_part = chunk[:split_idx]
        a_part = chunk[qa_sep_end:]

        question = q_part.replace("Question:", "").strip()
        answer = a_part.strip()

        is_last = (i == len(chunks) - 1)

        # user message
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        })

        # assistant message for in-context examples
        if (not is_last) and answer:
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            })

    return messages


class EvalModel(BaseEvalModel):
    """Eval wrapper for Qwen-2.5-VL (template).

    This implementation is intentionally generic and follows the same
    interface as the OpenFlamingo EvalModel used by `evaluate_vqa.py`.

    Requirements / assumptions when using a real Qwen-2.5-VL model:
    - The model repo supports `trust_remote_code=True` and implements a
      `generate(vision_x=..., input_ids=..., attention_mask=..., ...)`
      method compatible with the OpenFlamingo wrapper. If the model uses a
      different API, adjust `get_outputs` accordingly.
    - The tokenizer / model paths passed in `model_args` should point to
      the local HF-style directory (or HF repo id).

    Usage: call `evaluate_vqa.py --model qwen2_5vl --qwen_model_path <path> --qwen_tokenizer_path <path> --precision fp16 --device 0`
    """

    def __init__(self, model_args):
        # expected keys (leftovers from CLI are passed here)
        assert (
            "qwen_model_path" in model_args
            and "qwen_tokenizer_path" in model_args
            and "precision" in model_args
        ), "qwen2_5vl requires qwen_model_path, qwen_tokenizer_path and precision"

        # device
        self.device = (
            model_args["device"] if ("device" in model_args and int(model_args["device"]) >= 0) else "cpu"
        )

        self.autocast = get_autocast(model_args["precision"])
        self.cast_dtype = get_cast_dtype(model_args["precision"])

        if Qwen2_5_VLForConditionalGeneration is None or AutoTokenizer is None:
            raise ImportError(
                "transformers or torchvision not available in the environment. Please install them to use qwen2_5vl wrapper."
            )

        # load tokenizer, processor & model
        # trust_remote_code=True is often required for vendor-provided multimodal models
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_args["qwen_tokenizer_path"], trust_remote_code=True, use_fast=False
        )
        self.processor = AutoProcessor.from_pretrained(
            model_args["qwen_tokenizer_path"], trust_remote_code=True, use_fast=False
        )

        # If the model repo supports dtype on load, pass cast dtype
        load_kwargs = {"trust_remote_code": True}
        if self.cast_dtype is not None:
            load_kwargs["dtype"] = self.cast_dtype

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args["qwen_model_path"], **load_kwargs
        )

        # minimal image preprocessing: resize + center crop + to tensor + normalize
        # these values are generic; for best results, replace with the model's
        # own feature-extractor from HF hub.
        self.image_processor = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # move model to device
        self.model.to(self.device)
        self.model.eval()

        # prefer left padding for generation (many auto-regressive models expect this)
        try:
            self.tokenizer.padding_side = "left"
        except Exception:
            pass


    def get_outputs(
        self,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
        min_generation_length: int,
        max_generation_length: int,
        num_beams: int,
        length_penalty: float,
    ) -> List[str]:
        """
        Generate outputs for Flamingo-style multi-<image> VQA prompts
        using Qwen2.5-VL with chat-based in-context learning.
        """

        all_outputs = []
        #pdb.set_trace()
        for text_prompt, images in zip(batch_text, batch_images):
            # 1. Flamingo prompt -> Qwen chat messages
            messages = flamingo_prompt_to_qwen_messages(
                text_prompt,
                images,
            )
            # ✅ add system instruction (task-level, evaluation-safe)
            # messages = [{
            #     "role": "system",
            #     "content": [
            #         {
            #             "type": "text",
            #             "text": (
            #                 "Answer with a short answer only. "
            #             )
            #         }
            #     ],
            # }] + messages
            #pdb.set_trace()
            # 2. Apply Qwen chat template
            chat_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # 3. Processor (NOTE: images must be batch-wrapped)
            inputs = self.processor(
                text=[chat_text],
                images=[images],
                padding=True,
                return_tensors="pt",
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 4. Generate
            with torch.inference_mode(), self.autocast():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_generation_length,
                    min_new_tokens=min_generation_length,
                    num_beams=num_beams if num_beams > 1 else 1,
                    length_penalty=length_penalty if num_beams > 1 else 1.0,
                    do_sample=False,
                )
            # prompt 长度
            prompt_len = inputs["input_ids"].shape[1]

            # 只取新生成的 token
            answer_ids = generated_ids[:, prompt_len:]
            # 5. Decode (DO NOT manually trim input tokens)
            output_text = self.processor.batch_decode(
                answer_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )[0].strip()

            all_outputs.append(output_text)

        return all_outputs


    def get_vqa_prompt(self, question, answer=None) -> str:
        # keep same prompt format as OpenFlamingo wrapper for compatibility;
        # change if Qwen requires different image token markers.
        return f"<image>Question:{question} Short answer(a short answer only):{answer if answer is not None else ''}{'<|endofchunk|>' if answer is not None else ''}"

    def caption_prompt(self, caption=None) -> str:
        """Get the prompt to use for caption evaluation."""
        return f"<image>A picture of{caption if caption is not None else ''}{'<|endofchunk|>' if caption is not None else ''}"

    def classification_prompt(self, class_str=None) -> str:
        """Get the prompt to use for classification evaluation."""
        return f"<image>a photo of{class_str if class_str is not None else ''}{'<|endofchunk|>' if class_str is not None else ''}"

    def vqa_prompt(self, question, answer=None) -> str:
        """Alias for get_vqa_prompt for compatibility."""
        return self.get_vqa_prompt(question, answer)


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    elif precision == "fp32":
        cast_dtype = torch.float32
    return cast_dtype


def get_autocast(precision):
    if precision == "amp":
        return torch.cuda.amp.autocast
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress
