from rinalmo.data.alphabet import Alphabet
from rinalmo.data.constants import RNA_TOKENS
import torch

# Explicitly instantiate your tokenizer
alphabet = Alphabet(standard_tkns=RNA_TOKENS)

# Provide a representative RNA sequence from your actual dataset
example_sequence = "AUGCGAUCUAGCUAGCUAGC"  # Replace explicitly with your actual sequence if available

# Explicitly tokenize the sequence exactly as your pipeline currently does
tokenized_sequence = alphabet.tokenize(example_sequence)

# Print explicit tokenizer output for inspection
print("Original Sequence:", example_sequence)
print("Tokenized Output:", tokenized_sequence)
print("Unique token values:", set(tokenized_sequence))

# Convert explicitly to tensor to match your pipeline precisely
tokenized_tensor = torch.tensor(tokenized_sequence)
print("Tokenized Tensor Shape:", tokenized_tensor.shape)
print("Tokenized Tensor:", tokenized_tensor)
