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
from transformers.generation import (
    LogitsProcessorList,
    PrefixConstrainedLogitsProcessor,
    UnbatchedClassifierFreeGuidanceLogitsProcessor,
)
import torch
from cog import BasePredictor, Input, Path

from emu3.mllm.processing_emu3 import Emu3Processor


MODEL_CACHE = "model_cache"
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
            f"{MODEL_CACHE}/Emu3-Gen",  # "BAAI/Emu3-Gen"
            device_map="cuda:0",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            f"{MODEL_CACHE}/Emu3-Gen", trust_remote_code=True
        )  # "BAAI/Emu3-Gen"
        image_processor = AutoImageProcessor.from_pretrained(
            f"{MODEL_CACHE}/Emu3-VisionTokenizer", trust_remote_code=True
        )  # "BAAI/Emu3-VisionTokenizer"
        image_tokenizer = AutoModel.from_pretrained(
            f"{MODEL_CACHE}/Emu3-VisionTokenizer",
            device_map="cuda:0",
            trust_remote_code=True,
        ).eval()  # "BAAI/Emu3-VisionTokenizer"
        self.processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)

        self.kwargs = dict(
            mode="G",
            ratio="1:1",
            image_area=self.model.config.image_area,
            return_tensors="pt",
        )

        # prepare hyper parameters
        self.generation_config = GenerationConfig(
            use_cache=True,
            eos_token_id=self.model.config.eos_token_id,
            pad_token_id=self.model.config.pad_token_id,
            max_new_tokens=40960,
            do_sample=True,
            top_k=2048,
        )

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="a portrait of young girl.",
        ),
        positive_prompt: str = Input(
            default="masterpiece, film grained, best quality.",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default="lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry.",
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=3
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        pos_inputs = self.processor(
            text=prompt + " " + positive_prompt.strip(), **self.kwargs
        )
        neg_inputs = self.processor(text=negative_prompt, **self.kwargs)

        h, w = pos_inputs.image_size[0]
        constrained_fn = self.processor.build_prefix_constrained_fn(h, w)
        logits_processor = LogitsProcessorList(
            [
                UnbatchedClassifierFreeGuidanceLogitsProcessor(
                    guidance_scale,
                    self.model,
                    unconditional_ids=neg_inputs.input_ids.to("cuda:0"),
                ),
                PrefixConstrainedLogitsProcessor(
                    constrained_fn,
                    num_beams=1,
                ),
            ]
        )

        # generate
        outputs = self.model.generate(
            pos_inputs.input_ids.to("cuda:0"),
            self.generation_config,
            logits_processor=logits_processor,
        )

        out_path = "/tmp/out.png"

        mm_list = self.processor.decode(outputs[0])
        print(len(mm_list))
        print(mm_list)
        for idx, im in enumerate(mm_list):
            if not isinstance(im, Image.Image):
                continue
            im.save(out_path)
            return Path(out_path)
