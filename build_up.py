# build_up.py

import torch
from ocr import run_ocr_inference  # Import the function from ocr.py


def group_token_predictions(tokens, predicted_labels, label_map):
    """
    Groups contiguous tokens that have the same predicted label.

    Args:
        tokens (List[str]): The list of tokens.
        predicted_labels (List[int]): The list of predicted label IDs.
        label_map (dict): A mapping from label IDs to label names.

    Returns:
        grouped (List[tuple]): A list of tuples (label, text) where contiguous tokens with the same label are merged.
    """
    if not tokens or not predicted_labels:
        return []

    current_label = label_map.get(predicted_labels[0], "O")
    current_tokens = [tokens[0]]
    grouped = []

    for token, label in zip(tokens[1:], predicted_labels[1:]):
        label_text = label_map.get(label, "O")
        if label_text == current_label:
            current_tokens.append(token)
        else:
            grouped.append((current_label, " ".join(current_tokens)))
            current_label = label_text
            current_tokens = [token]

    grouped.append((current_label, " ".join(current_tokens)))
    return grouped


def build_structured_data(grouped_entities):
    """
    Converts a list of (label, text) tuples into a structured dictionary.

    Args:
        grouped_entities (List[tuple]): List of tuples in the form (label, text).

    Returns:
        structured_data (dict): A dictionary with entity labels as keys and concatenated texts as values.
    """
    structured_data = {}
    for label, text in grouped_entities:
        if label == "O":
            continue  # Skip tokens not marked as an entity.
        if label in structured_data:
            structured_data[label] += " " + text
        else:
            structured_data[label] = text
    return structured_data


def post_process_tokens(tokens, predicted_labels, label_map):
    """
    Post-processes token predictions by grouping contiguous tokens and then building structured data.

    Returns:
        grouped_entities: List of (label, text) tuples.
        structured_data: Dictionary mapping entity labels to concatenated text.
    """
    grouped_entities = group_token_predictions(
        tokens, predicted_labels, label_map)
    structured_data = build_structured_data(grouped_entities)
    return grouped_entities, structured_data


def main():
    # Call the OCR/inference function from ocr.py.
    # Update this path as needed.
    pdf_path = "bank_statements/commonwealthbank.pdf"
    tokens, predicted_labels = run_ocr_inference(pdf_path)

    # Define the label map (must be consistent with your model training or assumptions).
    label_map = {
        0: "O",
        1: "NAME",
        2: "ADDRESS",
        3: "SORT_CODE",
        4: "ACCOUNT_NUMBER",
        5: "TRANSACTION_ROW",
        6: "DATE",
        7: "AMOUNT"
    }

    # Post-process the token predictions.
    grouped_entities, structured_data = post_process_tokens(
        tokens, predicted_labels, label_map)

    # Display the results.
    print("Grouped Token Predictions:")
    for label, text in grouped_entities:
        print(f"{label}: {text}")

    print("\nStructured Data:")
    for field, value in structured_data.items():
        print(f"{field}: {value}")


if __name__ == '__main__':
    main()
