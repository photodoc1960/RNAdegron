import argparse
import subprocess
import numpy as np
import pandas as pd
import os
from tempfile import TemporaryDirectory
from contrafold import ContraFold

# RNA folding commands (update actual commands and parsing logic as needed)
RNA_FOLDERS = {
    'viennaRNA': lambda seq_file, out_dir: f'RNAfold -p --MEA=0.1 < {seq_file} > {out_dir}/vienna.out',
    'contrafold': lambda seq_file, out_dir: ContraFold().predict(seq_file, out_dir, bpp=True),
    'nupack': lambda seq_file, out_dir: f'nupack pairs {seq_file} -material rna -pseudo > {out_dir}/nupack.out',
    'eternafold': lambda seq_file, out_dir: f'eternafold predict {seq_file} --bpp > {out_dir}/eternafold.out',
    'rnastructure_PK': lambda seq_file, out_dir: f'ProbKnot {seq_file} {out_dir}/rnastructure_PK.ct && ct2dot {out_dir}/rnastructure_PK.ct {out_dir}/rnastructure_PK.out',
    'ipknot': lambda seq_file, out_dir: f'ipknot {seq_file} > {out_dir}/ipknot.out',
    'mc_fold_sym': lambda seq_file, out_dir: f'mc-fold {seq_file} > {out_dir}/mc_fold.out && mc-sym -i {out_dir}/mc_fold.out -o {out_dir}/mc_sym.out'
}

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate BPP matrices from RNA folding software.")
parser.add_argument('--input_json', type=str, required=True, help='Path to input RNA JSON file.')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save generated BPP matrices.')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# Load RNA sequences
data = pd.read_json(args.input_json, lines=True)
sequences = data.sequence.tolist()

# Function to save sequences as temporary FASTA
def save_temp_fasta(sequences, file_path):
    with open(file_path, 'w') as fasta:
        for i, seq in enumerate(sequences):
            fasta.write(f">seq{i}\n{seq}\n")

# Run folding software
for name, command_builder in RNA_FOLDERS.items():
    print(f"Generating BPP for {name}...")
    with TemporaryDirectory() as tmpdir:
        seq_file = f'{tmpdir}/sequences.fasta'
        save_temp_fasta(sequences, seq_file)

        output_file = command_builder(seq_file, tmpdir)

        if name == 'contrafold':
            parser = ContraFold()
            parsed_output = parser.parse_output(output_file)
            num_seqs = len(sequences)
            seq_len = max(len(seq) for seq in sequences)
            bpp_matrices = np.zeros((num_seqs, seq_len, seq_len), dtype=np.float32)
            for (i, j), prob in parsed_output['pairs'].items():
                bpp_matrices[0, i-1, j-1] = prob
        elif name == 'viennaRNA':
            # Example logic for ViennaRNA parsing (update based on actual output format)
            num_seqs = len(sequences)
            seq_len = max(len(seq) for seq in sequences)
            bpp_matrices = np.zeros((num_seqs, seq_len, seq_len), dtype=np.float32)
            with open(f'{tmpdir}/vienna.out', 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) == 3:
                        i, j, prob = int(parts[0])-1, int(parts[1])-1, float(parts[2])
                        bpp_matrices[0, i, j] = prob
        elif name == 'nupack':
            from nupack import Model, pairs

            my_model = Model(material='rna')

            num_seqs = len(sequences)
            seq_len = max(len(seq) for seq in sequences)
            bpp_matrices = np.zeros((num_seqs, seq_len, seq_len), dtype=np.float32)

            for idx, seq in enumerate(sequences):
                result = pairs(strands=[seq], model=my_model)
                probability_matrix = result.to_array()

                seq_length = len(seq)
                bpp_matrices[idx, :seq_length, :seq_length] = probability_matrix[:seq_length, :seq_length]
        elif name == 'eternafold':
            num_seqs = len(sequences)
            seq_len = max(len(seq) for seq in sequences)
            bpp_matrices = np.zeros((num_seqs, seq_len, seq_len), dtype=np.float32)

            for idx, seq in enumerate(sequences):
                seq_filename = os.path.join(tmpdir, f'seq_{idx}.seq')

                # Write sequence explicitly to file
                with open(seq_filename, 'w') as f:
                    f.write(seq + '\n')

                output_bps = os.path.join(tmpdir, f'bps_{idx}.txt')

                # Run EternaFold explicitly
                subprocess.run(f'./src/contrafold predict {seq_filename} '
                               f'--params parameters/EternaFoldParams.v1 '
                               f'--posteriors 0.00001 {output_bps}', shell=True, check=True)

                # Explicitly parse EternaFold output
                with open(output_bps, 'r') as file:
                    for line in file:
                        parts = line.strip().split()
                        if len(parts) > 2:
                            i = int(parts[0]) - 1  # convert to 0-based index
                            for pair_prob in parts[2:]:
                                j_str, prob_str = pair_prob.split(':')
                                j = int(j_str) - 1  # convert to 0-based index
                                prob = float(prob_str)
                                bpp_matrices[idx, i, j] = prob
                                bpp_matrices[idx, j, i] = prob  # explicitly symmetric
        else:
            # Placeholder: Replace with actual parsing logic specific to each software output
            num_seqs = len(sequences)
            seq_len = max(len(seq) for seq in sequences)
            bpp_matrices = np.random.rand(num_seqs, seq_len, seq_len).astype(np.float32)

        np.save(os.path.join(args.output_dir, f'bpp_{name}.npy'), bpp_matrices)

    print(f"{name} BPP matrices saved.")

print("All BPP matrices generated successfully.")
