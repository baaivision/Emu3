# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import time
import subprocess
from PIL import Image
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoImageProcessor,
    AutoModelForCausalLM,
)
from transformers.generation.configuration_utils import GenerationConfig
import torch
from cog import BasePredictor, Input, Path

from emu3.mllm.processing_emu3 import Emu3Processor


MODEL_CACHE = "model_cache_chat"
MODEL_URL = (
    f"https://weights.replicate.delivery/default/baaivision/Emu3/{MODEL_CACHE}.tar"
)
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE

TORCH_TYPE = torch.bfloat16
DEVICE = "cuda:0"


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        # prepare model and processor
        self.model = AutoModelForCausalLM.from_pretrained(
            f"{MODEL_CACHE}/Emu3-Chat",  # "BAAI/Emu3-Chat"
            device_map="cuda:0",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            f"{MODEL_CACHE}/Emu3-Chat", trust_remote_code=True
        )  # "BAAI/Emu3-Chat"
        image_processor = AutoImageProcessor.from_pretrained(
            f"{MODEL_CACHE}/Emu3-VisionTokenizer", trust_remote_code=True
        )  # "BAAI/Emu3-VisionTokenizer"
        image_tokenizer = AutoModel.from_pretrained(
            f"{MODEL_CACHE}/Emu3-VisionTokenizer",
            device_map="cuda:0",
            trust_remote_code=True,
        ).eval()  # "BAAI/Emu3-VisionTokenizer"
        self.processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)
        # prepare hyper parameters
        self.generation_config = GenerationConfig(
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    def predict(
        self,
        text: str = Input(
            description="Input prompt",
            default="Please describe the image.",
        ),
        image: Path = Input(
            default="Input image",
        ),
        temperature: float = Input(
            description="Controls randomness. Lower values make the model more deterministic, higher values make it more random.",
            default=0.7,
            ge=0.0,
            le=1.0,
        ),
        top_p: float = Input(
            description="Controls diversity of the output. Valid when temperature > 0. Lower values make the output more focused, higher values make it more diverse.",
            default=0.9,
            ge=0.0,
            le=1.0,
        ),
        max_new_tokens: int = Input(
            description="Maximum number of tokens to generate", default=256, ge=1
        ),
    ) -> str:
        """Run a single prediction on the model"""

        img = Image.open(str(image))

        inputs = self.processor(
            text=text,
            image=img,
            mode="U",
            padding_side="left",
            padding="longest",
            return_tensors="pt",
        )

        outputs = self.model.generate(
            inputs.input_ids.to("cuda:0"),
            self.generation_config,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        outputs = outputs[:, inputs.input_ids.shape[-1] :]
        return self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
