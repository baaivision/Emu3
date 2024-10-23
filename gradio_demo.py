# -*- coding: utf-8 -*-

import base64
import io
from PIL import Image

import gradio as gr
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoImageProcessor,
    AutoModel,
)
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation import (
    LogitsProcessorList,
    PrefixConstrainedLogitsProcessor,
    UnbatchedClassifierFreeGuidanceLogitsProcessor,
)
import torch

from emu3.mllm.processing_emu3 import Emu3Processor

def image2str(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    i_str = base64.b64encode(buf.getvalue()).decode()
    return f'<div style="float:left"><img src="data:image/png;base64, {i_str}"></div>'

device = "cuda" if torch.cuda.is_available() else "cpu"

# Model paths
EMU_GEN_HUB = "BAAI/Emu3-Gen"
EMU_CHAT_HUB = "BAAI/Emu3-Chat"
VQ_HUB = "BAAI/Emu3-VisionTokenizer"

# Prepare models and processors
gen_model = AutoModelForCausalLM.from_pretrained(
    EMU_GEN_HUB,
    device_map="cpu",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    trust_remote_code=True,
).eval()

chat_model = AutoModelForCausalLM.from_pretrained(
    EMU_CHAT_HUB,
    device_map="cpu",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    trust_remote_code=True,
).eval()

tokenizer = AutoTokenizer.from_pretrained(
    EMU_CHAT_HUB, trust_remote_code=True, padding_side="left",
)
image_processor = AutoImageProcessor.from_pretrained(
    VQ_HUB, trust_remote_code=True,
)
image_tokenizer = AutoModel.from_pretrained(
    VQ_HUB, device_map="cpu", trust_remote_code=True,
).eval()

image_tokenizer.to(device)

processor = Emu3Processor(
    image_processor, image_tokenizer, tokenizer
)

def generate_image(prompt):
    POSITIVE_PROMPT = " masterpiece, film grained, best quality."
    NEGATIVE_PROMPT = (
        "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, "
        "fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, "
        "signature, watermark, username, blurry."
    )

    classifier_free_guidance = 3.0
    full_prompt = prompt + POSITIVE_PROMPT

    kwargs = dict(
        mode="G",
        ratio="1:1",
        image_area=gen_model.config.image_area,
        return_tensors="pt",
    )
    pos_inputs = processor(text=full_prompt, **kwargs)
    neg_inputs = processor(text=NEGATIVE_PROMPT, **kwargs)

    # Prepare hyperparameters
    GENERATION_CONFIG = GenerationConfig(
        use_cache=True,
        eos_token_id=gen_model.config.eos_token_id,
        pad_token_id=gen_model.config.pad_token_id,
        max_new_tokens=40960,
        do_sample=True,
        top_k=2048,
    )

    torch.cuda.empty_cache()
    gen_model.to(device)

    h = pos_inputs.image_size[:, 0]
    w = pos_inputs.image_size[:, 1]
    constrained_fn = processor.build_prefix_constrained_fn(h, w)
    logits_processor = LogitsProcessorList([
        UnbatchedClassifierFreeGuidanceLogitsProcessor(
            classifier_free_guidance,
            gen_model,
            unconditional_ids=neg_inputs.input_ids.to(device),
        ),
        PrefixConstrainedLogitsProcessor(
            constrained_fn,
            num_beams=1,
        ),
    ])

    # Generate
    outputs = gen_model.generate(
        pos_inputs.input_ids.to(device),
        generation_config=GENERATION_CONFIG,
        logits_processor=logits_processor,
        attention_mask=pos_inputs.attention_mask.to(device),
    )

    mm_list = processor.decode(outputs[0])
    result = None
    for idx, im in enumerate(mm_list):
        if isinstance(im, Image.Image):
            result = im
            break

    gen_model.cpu()
    torch.cuda.empty_cache()
    
    return result

def vision_language_understanding(image, text):
    inputs = processor(
        text=text,
        image=image,
        mode="U",
        padding="longest",
        return_tensors="pt",
    )

    # Prepare hyperparameters
    GENERATION_CONFIG = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=1024,
    )

    torch.cuda.empty_cache()
    chat_model.to(device)

    # Generate
    outputs = chat_model.generate(
        inputs.input_ids.to(device),
        generation_config=GENERATION_CONFIG,
        attention_mask=inputs.attention_mask.to(device),
    )

    outputs = outputs[:, inputs.input_ids.shape[-1] :]
    response = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    chat_model.cpu()
    torch.cuda.empty_cache()
    
    return response

    
def chat(history, user_input, user_image):
    if user_image is not None:
        # Use Emu3-Chat for vision-language understanding
        response = vision_language_understanding(user_image, user_input)
        # Append the user input and response to the history
        history = history + [(image2str(user_image) + "<br>" + user_input, response)]
    else:
        # Use Emu3-Gen for image generation
        generated_image = generate_image(user_input)
        if generated_image is not None:
            # Append the user input and generated image to the history
            history = history + [(user_input, image2str(generated_image))]
        else:
            # If image generation failed, respond with an error message
            history = history + [
                (user_input, "Sorry, I could not generate an image.")
            ]

    return history, history, gr.update(value=None)

    
def clear_input():
    return gr.update(value="")

    
with gr.Blocks() as demo:
    gr.Markdown("# Emu3 Chatbot Demo")
    gr.Markdown(
        "This is a chatbot demo for image generation and vision-language understanding using Emu3 models."
    )
    gr.Markdown(
        "Please provide <b>only text input</b> for image generation (<b>\~600s</b>) and <b>both image and text</b> for vision-language understanding (<b>\~20s</b>)"
    )

    state = gr.State([])
    with gr.Row():
        with gr.Column(scale=0.2):
            user_input = gr.Textbox(
                show_label=False, placeholder="Type your message here...", lines=10, container=False,
            )
            user_image = gr.Image(
                sources="upload", type="pil", label="Upload an image (optional)"
            )
            submit_btn = gr.Button("Send")

        with gr.Column(scale=0.8):
            chatbot = gr.Chatbot(height=800)

    submit_btn.click(
        chat,
        inputs=[state, user_input, user_image],
        outputs=[chatbot, state, user_image],
    ).then(fn=clear_input, inputs=[], outputs=user_input, queue=False)
    user_input.submit(
        chat,
        inputs=[state, user_input, user_image],
        outputs=[chatbot, state, user_image],
    ).then(fn=clear_input, inputs=[], outputs=user_input, queue=False)

demo.launch(max_threads=1).queue()
