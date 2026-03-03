from deepchem.feat.dnabert_tokenizer import DNABertTokenizer

sequence = "ACGTACGTACGT"

tokenizer = DNABertTokenizer.from_pretrained(
    "zhihan1996/DNABERT-2-117M",
    trust_remote_code=True
)

output = tokenizer(sequence)

print("Sequence:")
print(sequence)

print("\nFull Output:")
print(output)

print("\nInput IDs:")
print(output["input_ids"])

print("\nAttention Mask:")
print(output["attention_mask"])

print("\nTokens:")
print(tokenizer.convert_ids_to_tokens(output["input_ids"]))

print("\nDecoded:")
print(tokenizer.decode(output["input_ids"]))