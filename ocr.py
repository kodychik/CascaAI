# ocr.py

import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import torch
from transformers import LayoutLMv2Processor, LayoutLMv2ForTokenClassification


def pdf_to_images(pdf_path):
    """
    Converts each page of a PDF into a PIL Image.
    """
    images = convert_from_path(pdf_path)
    return images


def ocr_with_layout(image):
    """
    Uses Tesseract to extract words and bounding boxes from an image.

    Returns:
        words: List of recognized words.
        boxes: List of bounding boxes in [left, top, right, bottom] format.
    """
    data = pytesseract.image_to_data(
        image, output_type=pytesseract.Output.DICT)
    words = []
    boxes = []
    for i in range(len(data['text'])):
        word = data['text'][i].strip()
        if word:
            words.append(word)
            left, top, width, height = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            boxes.append([left, top, left + width, top + height])
    return words, boxes


def normalize_boxes(boxes, width, height):
    """
    Normalizes bounding boxes to the 0-1000 range as expected by LayoutLMv2.

    Args:
        boxes: List of boxes in pixel coordinates [left, top, right, bottom].
        width: Width of the image in pixels.
        height: Height of the image in pixels.

    Returns:
        normalized_boxes: List of boxes with coordinates scaled to 0-1000.
    """
    normalized_boxes = []
    for box in boxes:
        x0, y0, x1, y1 = box
        normalized_box = [
            int(1000 * (x0 / width)),
            int(1000 * (y0 / height)),
            int(1000 * (x1 / width)),
            int(1000 * (y1 / height))
        ]
        normalized_boxes.append(normalized_box)
    return normalized_boxes


def prepare_layoutlm_inputs(image, words, boxes, processor):
    """
    Prepares inputs for LayoutLMv2 using the provided image, words, and bounding boxes.
    """
    encoding = processor(image, words, boxes=boxes,
                         return_tensors="pt", truncation=True)
    return encoding


def run_ocr_inference(pdf_path):
    """
    Processes the PDF at pdf_path and runs OCR + LayoutLMv2 inference on the first page.

    Returns:
        tokens: List of tokens (strings) output by the processor.
        predicted_labels: List of predicted label IDs (integers) for each token.
    """
    images = pdf_to_images(pdf_path)
    # For demonstration, process only the first page.
    image = images[0]
    words, boxes = ocr_with_layout(image)
    width, height = image.size
    normalized_boxes = normalize_boxes(boxes, width, height)

    # Initialize the processor (with apply_ocr=False because we supply our own OCR results).
    processor = LayoutLMv2Processor.from_pretrained(
        "microsoft/layoutlmv2-base-uncased", revision="no_ocr", apply_ocr=False
    )

    # Load the model (using PyTorch); note that we use num_labels=8 for this example.
    model = LayoutLMv2ForTokenClassification.from_pretrained(
        "microsoft/layoutlmv2-base-uncased", revision="no_ocr", num_labels=8
    )

    # Prepare inputs
    encoding = prepare_layoutlm_inputs(
        image, words, normalized_boxes, processor)

    # Run inference
    outputs = model(**encoding)
    logits = outputs.logits
    predicted_labels = torch.argmax(logits, dim=2).squeeze().tolist()

    # Convert input IDs back to tokens
    tokens = processor.tokenizer.convert_ids_to_tokens(
        encoding["input_ids"].squeeze().tolist())

    return tokens, predicted_labels


# Optional: for quick testing you can add a main guard.
if __name__ == '__main__':
    pdf_path = "bank_statements/commonwealthbank.pdf"  # Update path as needed.
    tokens, predicted_labels = run_ocr_inference(pdf_path)
    # Print token predictions for a quick check:
    label_map = {
        0: "O", 1: "NAME", 2: "ADDRESS", 3: "SORT_CODE",
        4: "ACCOUNT_NUMBER", 5: "TRANSACTION_ROW", 6: "DATE", 7: "AMOUNT"
    }
    for token, label_id in zip(tokens, predicted_labels):
        print(f"{token} -> {label_map.get(label_id, 'O')}")
