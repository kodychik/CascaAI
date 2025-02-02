import re
import torch
from pdf2image import convert_from_path
from transformers import DonutProcessor, VisionEncoderDecoderModel


def main():
    # Load the Donut processor and model.
    # Ensure that there is no local folder named "naver-clova-ix/donut-base-finetuned-cord-v2"
    # so that the model is loaded from the Hugging Face Hub.
    processor = DonutProcessor.from_pretrained(
        "naver-clova-ix/donut-base-finetuned-cord-v2")
    model = VisionEncoderDecoderModel.from_pretrained(
        "naver-clova-ix/donut-base-finetuned-cord-v2")

    # Use CUDA if available.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Load the document image from a PDF.
    # Update with your PDF file path.
    pdf_path = "bank_statements/LloydsUK.pdf"
    images = convert_from_path(pdf_path)
    if not images:
        raise ValueError("No images were extracted from the PDF.")
    # Use the first page of the PDF.
    image = images[0]

    # Prepare decoder inputs.
    # Use a non-empty prompt and include special tokens.
    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(
        task_prompt, add_special_tokens=True, return_tensors="pt"
    ).input_ids

    # Prepare the image: set legacy=False to use the new behavior.
    pixel_values = processor(image, return_tensors="pt",
                             legacy=False).pixel_values
    print("Pixel values shape:", pixel_values.shape)

    # Generate outputs.
    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    # Decode and clean the generated sequence.
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(
        processor.tokenizer.pad_token, "")
    # Remove the first task start token if present.
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
    print("Raw generated sequence:")
    print(sequence)

    # Convert the sequence to JSON.
    result_json = processor.token2json(sequence)
    print("\nStructured JSON:")
    print(result_json)


if __name__ == "__main__":
    main()
