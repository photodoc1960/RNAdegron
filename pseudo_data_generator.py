# Create synthetic_data_generation.py script
from synthetic_data_generator import RNASyntheticDataGenerator

# Generate synthetic dataset
generator = RNASyntheticDataGenerator()
synthetic_df = generator.generate_synthetic_dataset(
    n_sequences=10000,  # Start with 5K sequences
    output_path='data/synthetic_rna_dataset.pkl'
)
print(f"Generated {len(synthetic_df)} synthetic sequences")