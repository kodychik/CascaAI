import torch
from transformers import AutoTokenizer, AutoModel

import numpy as np
import torch
import torchvision.transforms as T
#from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import os
import pdfplumber

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
device = "cuda" if torch.cuda.is_available() else "cpu"

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size),
                 interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image, input_size=448, max_num=12):
    #image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values).to(torch.bfloat16)
    return pixel_values.to(device)


def pdf_to_images(pdf_path, dpi=300):
    """
    Converts each page of a PDF into a JPG image and stores them in a list.

    Args:
        pdf_path (str): Path to the input PDF file.
        dpi (int, optional): Dots per inch for rendering the images. Defaults to 300.

    Returns:
        list: A list of PIL Image objects, each representing a page in the PDF.
    """
    images = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Render the PDF page as an image
            image = page.to_image(resolution=dpi).original

            # Convert to RGB to ensure compatibility
            image = image.convert("RGB")

            # Append the image to the list
            images.append(image)

    return images


def inference():

    # path = "OpenGVLab/InternVL2_5-8B"
    # device_map = split_model('InternVL2_5-8B')
    # model = AutoModel.from_pretrained(
    #     path,
    #     torch_dtype=torch.bfloat16,
    #     low_cpu_mem_usage=True,
    #     use_flash_attn=True,
    #     trust_remote_code=True,
    #     device_map=device_map).eval()

    
    print(f"Using device: {device}")

    path = 'OpenGVLab/InternVL2_5-2B'
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        #load_in_8bit=True,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, use_fast=False)

    images = pdf_to_images("bank_statements/LloydsUK.pdf")
    # set the max number of tiles in `max_num`
    print("@@@@@@@@")
    pixel_values = load_image(images[0], max_num=12).to(torch.bfloat16).to(device)

    generation_config = dict(max_new_tokens=1024, do_sample=False)

    # single-image single-round conversation (单图单轮对话)
    print("^^^^^^^^^")
    question = '<image>\nThis is a bank statement, Obtain the text and table information. Output in JSON format.'
    response = model.chat(tokenizer, pixel_values, question, generation_config)
    print("*****")
    print(f'User: {question}\nAssistant: {response}')


if __name__=="__main__":
    inference()