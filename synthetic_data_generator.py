# ADVANCED STRATEGY 2: Random Sequence Generation for Pseudo-Labeling
# Add this module as synthetic_data_generator.py

import numpy as np
import pandas as pd
import RNA
import os
import pickle
from tqdm import tqdm
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


class RNASyntheticDataGenerator:
    """
    Generate synthetic RNA sequences with computed structural properties
    for unlimited pseudo-labeling data augmentation.
    """

    def __init__(self, base_composition=None, gc_content_range=(0.3, 0.7)):
        """
        Initialize synthetic data generator.

        Args:
            base_composition: Dict with base frequencies (default: uniform)
            gc_content_range: Tuple of (min, max) GC content
        """
        self.base_composition = base_composition or {'A': 0.25, 'C': 0.25, 'G': 0.25, 'U': 0.25}
        self.gc_content_range = gc_content_range

    def generate_sequence_with_gc_constraint(self, length, target_gc=None):
        """
        Generate RNA sequence with specified GC content constraint.

        Args:
            length: Target sequence length
            target_gc: Target GC content (if None, random within range)

        Returns:
            Generated RNA sequence string
        """
        if target_gc is None:
            target_gc = np.random.uniform(*self.gc_content_range)

        # Calculate target nucleotide counts
        target_gc_count = int(length * target_gc)
        target_au_count = length - target_gc_count

        # Distribute GC and AU nucleotides
        c_count = np.random.binomial(target_gc_count, 0.5)
        g_count = target_gc_count - c_count
        a_count = np.random.binomial(target_au_count, 0.5)
        u_count = target_au_count - a_count

        # Generate sequence
        nucleotides = ['A'] * a_count + ['C'] * c_count + ['G'] * g_count + ['U'] * u_count
        np.random.shuffle(nucleotides)

        return ''.join(nucleotides)

    def compute_sequence_properties(self, sequence):
        """
        Compute comprehensive structural properties for RNA sequence.

        Args:
            sequence: RNA sequence string

        Returns:
            Dictionary with structural properties
        """
        try:
            # ViennaRNA computations
            fc = RNA.fold_compound(sequence)

            # MFE structure and energy
            mfe_structure, mfe_energy = RNA.fold(sequence)

            # Partition function and ensemble properties
            fc.pf()
            ensemble_energy = fc.pf()[1]

            # Base pairing probability matrix
            bpp_dict = fc.bpp()
            L = len(sequence)
            bpp_matrix = np.zeros((L, L))

            for i in range(1, L + 1):
                for j in range(i + 1, L + 1):
                    if (i, j) in bpp_dict:
                        bpp_matrix[i - 1, j - 1] = bpp_dict[(i, j)]
                        bpp_matrix[j - 1, i - 1] = bpp_dict[(i, j)]

            # Loop type annotation
            loop_types = self._annotate_loop_types(sequence, mfe_structure)

            # Structural features
            paired_positions = set()
            for i, char in enumerate(mfe_structure):
                if char in '()':
                    paired_positions.add(i)

            # Compute distance features
            nearest_paired = self._compute_nearest_paired_distances(sequence, paired_positions)
            nearest_unpaired = self._compute_nearest_unpaired_distances(sequence, paired_positions)
            graph_distances = self._compute_graph_distances(sequence, mfe_structure)

            return {
                'sequence': sequence,
                'mfe_structure': mfe_structure,
                'mfe_energy': mfe_energy,
                'ensemble_energy': ensemble_energy,
                'bpp_matrix': bpp_matrix,
                'loop_types': loop_types,
                'nearest_paired': nearest_paired,
                'nearest_unpaired': nearest_unpaired,
                'graph_distances': graph_distances,
                'gc_content': (sequence.count('G') + sequence.count('C')) / len(sequence)
            }

        except Exception as e:
            print(f"Error computing properties for sequence: {e}")
            return None

    def _annotate_loop_types(self, sequence, structure):
        """Annotate loop types using bracket structure."""
        loop_types = ['E'] * len(sequence)  # Default: exterior
        stack = []

        for i, char in enumerate(structure):
            if char == '(':
                stack.append(i)
                loop_types[i] = 'S'  # Stem
            elif char == ')':
                if stack:
                    j = stack.pop()
                    loop_types[i] = 'S'  # Stem

                    # Classify enclosed region
                    if structure[j + 1:i].count('(') == 0:
                        # Hairpin loop
                        for k in range(j + 1, i):
                            loop_types[k] = 'H'
                    else:
                        # Internal or multi-loop classification
                        unpaired_regions = 0
                        in_unpaired = False
                        for k in range(j + 1, i):
                            if structure[k] == '.' and not in_unpaired:
                                unpaired_regions += 1
                                in_unpaired = True
                            elif structure[k] != '.' and in_unpaired:
                                in_unpaired = False

                        loop_type = 'M' if unpaired_regions > 2 else 'I'
                        for k in range(j + 1, i):
                            if structure[k] == '.':
                                loop_types[k] = loop_type

        return ''.join(loop_types)

    def _compute_nearest_paired_distances(self, sequence, paired_positions):
        """Compute distance to nearest paired position for each nucleotide."""
        L = len(sequence)
        distances = np.full(L, L, dtype=float)  # Initialize with max distance

        paired_array = np.array(sorted(paired_positions))
        if len(paired_array) == 0:
            return distances

        for i in range(L):
            if i in paired_positions:
                distances[i] = 0
            else:
                # Find nearest paired position
                closest_idx = np.searchsorted(paired_array, i)
                candidates = []

                if closest_idx < len(paired_array):
                    candidates.append(abs(i - paired_array[closest_idx]))
                if closest_idx > 0:
                    candidates.append(abs(i - paired_array[closest_idx - 1]))

                if candidates:
                    distances[i] = min(candidates)

        return distances

    def _compute_nearest_unpaired_distances(self, sequence, paired_positions):
        """Compute distance to nearest unpaired position for each nucleotide."""
        L = len(sequence)
        unpaired_positions = set(range(L)) - paired_positions
        distances = np.full(L, L, dtype=float)

        unpaired_array = np.array(sorted(unpaired_positions))
        if len(unpaired_array) == 0:
            return distances

        for i in range(L):
            if i in unpaired_positions:
                distances[i] = 0
            else:
                # Find nearest unpaired position
                closest_idx = np.searchsorted(unpaired_array, i)
                candidates = []

                if closest_idx < len(unpaired_array):
                    candidates.append(abs(i - unpaired_array[closest_idx]))
                if closest_idx > 0:
                    candidates.append(abs(i - unpaired_array[closest_idx - 1]))

                if candidates:
                    distances[i] = min(candidates)

        return distances

    def _compute_graph_distances(self, sequence, structure):
        """Compute graph-based distances considering base pairing topology."""
        L = len(sequence)
        distances = np.full((L, L), L, dtype=float)

        # Build adjacency list from structure
        pairs = {}
        stack = []

        for i, char in enumerate(structure):
            if char == '(':
                stack.append(i)
            elif char == ')' and stack:
                j = stack.pop()
                pairs[i] = j
                pairs[j] = i

        # Compute shortest paths in the structure graph
        for i in range(L):
            distances[i, i] = 0
            visited = set([i])
            queue = [(i, 0)]

            while queue:
                pos, dist = queue.pop(0)

                # Adjacent positions in sequence
                for next_pos in [pos - 1, pos + 1]:
                    if 0 <= next_pos < L and next_pos not in visited:
                        distances[i, next_pos] = dist + 1
                        visited.add(next_pos)
                        queue.append((next_pos, dist + 1))

                # Base paired position
                if pos in pairs:
                    pair_pos = pairs[pos]
                    if pair_pos not in visited:
                        distances[i, pair_pos] = dist + 1
                        visited.add(pair_pos)
                        queue.append((pair_pos, dist + 1))

        return distances

    def generate_synthetic_dataset(self, n_sequences, length_distribution=None,
                                   output_path='synthetic_rna_dataset.pkl', n_workers=None):
        """
        Generate comprehensive synthetic RNA dataset.

        Args:
            n_sequences: Number of sequences to generate
            length_distribution: Dict with length frequencies (default: 107, 130 bp)
            output_path: Path to save generated dataset
            n_workers: Number of parallel workers (default: CPU count)

        Returns:
            DataFrame with synthetic sequences and properties
        """
        if length_distribution is None:
            length_distribution = {107: 0.17, 130: 0.83}  # Based on competition data

        if n_workers is None:
            n_workers = min(multiprocessing.cpu_count(), 8)

        print(f"Generating {n_sequences} synthetic RNA sequences using {n_workers} workers...")

        # Generate sequence specifications
        sequence_specs = []
        for _ in range(n_sequences):
            length = np.random.choice(
                list(length_distribution.keys()),
                p=list(length_distribution.values())
            )
            target_gc = np.random.uniform(*self.gc_content_range)
            sequence_specs.append((length, target_gc))

        # Parallel generation with progress tracking
        synthetic_data = []

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all jobs
            future_to_spec = {
                executor.submit(self._generate_single_sequence, spec): spec
                for spec in sequence_specs
            }

            # Collect results with progress bar
            for future in tqdm(as_completed(future_to_spec), total=n_sequences,
                               desc="Generating sequences"):
                try:
                    result = future.result()
                    if result is not None:
                        synthetic_data.append(result)
                except Exception as e:
                    print(f"Error generating sequence: {e}")

        # Convert to DataFrame
        synthetic_df = pd.DataFrame(synthetic_data)

        # Save dataset
        with open(output_path, 'wb') as f:
            pickle.dump(synthetic_df, f)

        print(f"Generated {len(synthetic_df)} synthetic sequences")
        print(f"Dataset saved to: {output_path}")
        print(f"GC content range: {synthetic_df['gc_content'].min():.3f} - {synthetic_df['gc_content'].max():.3f}")

        return synthetic_df

    def _generate_single_sequence(self, spec):
        """Generate single sequence with properties (for parallel execution)."""
        length, target_gc = spec
        sequence = self.generate_sequence_with_gc_constraint(length, target_gc)
        properties = self.compute_sequence_properties(sequence)

        if properties is not None:
            # Add synthetic ID
            properties['id'] = f"synthetic_{hash(sequence) % 1000000:06d}"
            properties['synthetic'] = True
            properties['length'] = length

        return properties


# INTEGRATION: Enhance pseudo-labeling pipeline
def integrate_synthetic_sequences_into_pseudolabeling(synthetic_dataset_path,
                                                      fold_models, device,
                                                      output_path='enhanced_pseudo_labels.pkl'):
    """
    Generate pseudo-labels for synthetic sequences using ensemble models.

    Args:
        synthetic_dataset_path: Path to synthetic sequence dataset
        fold_models: List of trained models for ensemble prediction
        device: Computation device
        output_path: Path to save enhanced pseudo-labels

    Returns:
        Dictionary with synthetic pseudo-labels and metadata
    """
    print("Loading synthetic dataset for pseudo-labeling...")
    with open(synthetic_dataset_path, 'rb') as f:
        synthetic_df = pickle.load(f)

    # Convert to format compatible with existing pipeline
    synthetic_sequences = synthetic_df['sequence'].tolist()
    synthetic_ids = synthetic_df['id'].tolist()

    print(f"Generating pseudo-labels for {len(synthetic_sequences)} synthetic sequences...")

    # Use existing pseudo-labeling infrastructure with synthetic data
    # This would integrate with your process_sequence_batch_with_structural_features function

    pseudo_predictions = []
    prediction_uncertainties = []

    # Implementation would follow your existing pseudo-labeling logic
    # but applied to synthetic sequences with their computed structural properties

    enhanced_pseudo_data = {
        'synthetic_sequences': synthetic_sequences,
        'synthetic_ids': synthetic_ids,
        'pseudo_predictions': pseudo_predictions,
        'prediction_uncertainties': prediction_uncertainties,
        'structural_properties': synthetic_df.to_dict('records'),
        'generation_metadata': {
            'n_sequences': len(synthetic_sequences),
            'length_distribution': synthetic_df['length'].value_counts().to_dict(),
            'gc_content_stats': {
                'mean': synthetic_df['gc_content'].mean(),
                'std': synthetic_df['gc_content'].std(),
                'range': (synthetic_df['gc_content'].min(), synthetic_df['gc_content'].max())
            }
        }
    }

    with open(output_path, 'wb') as f:
        pickle.dump(enhanced_pseudo_data, f)

    print(f"Enhanced pseudo-labels saved to: {output_path}")
    return enhanced_pseudo_data