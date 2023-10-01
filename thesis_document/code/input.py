text_encodings = tokenizer(text, ...)

return {
    "input_ids": text_encodings["input_ids"],
    "attention_mask": text_encodings["attention_mask"],
    "label": ...
}
