from transformers import LayoutLMv2Processor, LayoutLMv2ForTokenClassification
import torch
import ocr

# Load a pre-trained processor and model
processor = LayoutLMv2Processor.from_pretrained(
    "microsoft/layoutlmv2-base-uncased")
# For demonstration, we assume 7 labels, e.g., O, NAME, ADDRESS, SORT_CODE, ACCOUNT_NUMBER, TRANSACTION_ROW, DATE, etc.
# You must fine-tune the head on your own labeled dataset. Here we use a placeholder.
model = LayoutLMv2ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv2-base-uncased", num_labels=7)


def prepare_layoutlm_inputs(image, words, boxes):
    # Processor will resize image and normalize bounding boxes for you
    encoding = processor(image, words, boxes=boxes,
                         return_tensors="pt", truncation=True)
    return encoding


# Process first page of the PDF as an example
image = images[0]  # PIL image from pdf2image
words, boxes = ocr_with_layout(image)
encoding = prepare_layoutlm_inputs(image, words, boxes)
