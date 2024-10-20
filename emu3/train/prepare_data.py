# -*- coding: utf-8 -*-

import argparse
import json
import os

from PIL import Image
import torch

from emu3.tokenizer import Emu3VisionVQModel, Emu3VisionVQImageProcessor


def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, help='vision tokenizer path')
    parser.add_argument('--data-path', type=str, help='data path')
    parser.add_argument('--output-path', type=str, help='tokenized data save path')
    parser.add_argument('--image-area', type=int, default=720 * 720)

    args = parser.parse_args()
    return args


def smart_resize(image, image_area: int = 720 * 720):
    w, h = image.size
    current_area = h * w
    target_ratio = (image_area / current_area) ** 0.5

    th = int(round(h * target_ratio))
    tw = int(round(w * target_ratio))

    image = image.resize((tw, th))
    return image


def main():
    args = prepare_args()

    image_processor = Emu3VisionVQImageProcessor.from_pretrained(args.model_path)
    image_tokenizer = Emu3VisionVQModel.from_pretrained(args.model_path, device_map="cuda:0")
    image_tokenizer.eval()

    os.makedirs(f"{args.output_path}/feature", exist_ok=True)
    os.makedirs(f"{args.output_path}/list", exist_ok=True)

    datalist = {
        "prefix": f"{args.output_path}/feature",
        "path_list": []
    }

    with open(args.data_path) as f:
        input_data = json.load(f)

    for inp in input_data:
        name = inp["name"]
        prompt = inp["text"]

        image = Image.open(inp["image"]).convert("RGB")
        image = smart_resize(image, args.image_area)

        image = image_processor(image, return_tensors="pt")["pixel_values"]
        with torch.no_grad():
            image = image.cuda()
            token_ids = image_tokenizer.encode(image)

        token_ids = token_ids.squeeze(0).cpu().numpy()
        data = {
            "name": name,
            "images": token_ids,
            "texts": prompt
        }

        torch.save(data, f"{args.output_path}/feature/{name}.pth")
        datalist["path_list"].append(f"{name}.pth")

    with open(f"{args.output_path}/list/train.json", 'w') as f:
        json.dump(datalist, f)


if __name__ == "__main__":
    main()
