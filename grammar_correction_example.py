# based on the example from the HF model page: https://huggingface.co/Unbabel/gec-t5_small

from transformers import T5ForConditionalGeneration, T5Tokenizer

# takes a few minutes to download the models.
model = T5ForConditionalGeneration.from_pretrained("Unbabel/gec-t5_small")
tokenizer = T5Tokenizer.from_pretrained('t5-small')

sentence = "I like to swimming"
# Tokenization - breaking the sentence to "tokens" - words or word pieces
tokenized_sentence = tokenizer('gec: ' + sentence, max_length=128, truncation=True, padding='max_length', return_tensors='pt')

# The tokenized sentence contains an "input_ids" part - which contains token numbers, and an "attention_mask" part.
# The attention is used to decode different things - in this case, which tokens are the original sentence, and which are just padding.

# see the tokenized sentence
# tokenizer.decode(tokenized_sentence['input_ids'][0])

# pass the tokenized sentence through the model. T5 is a text-to-text model.
output = model.generate(
        input_ids=tokenized_sentence.input_ids,
        attention_mask=tokenized_sentence.attention_mask,
        max_length=128,
        num_beams=5,
        early_stopping=True,
    )

# Decode back from token ids to words.
corrected_sentence = tokenizer.decode(
    output[0],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True
)

print(corrected_sentence)
