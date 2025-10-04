import os
import subprocess
import pathlib
from typing import Optional, List, Dict, Union, Tuple


class ContraFold:
    """Interface for ContraFold RNA structure prediction."""

    def __init__(self, executable_path: Optional[str] = None):
        """
        Initialize ContraFold interface.

        Args:
            executable_path: Absolute path to contrafold executable.
                             If None, attempts to locate in PATH or current directory.
        """
        self.executable = self._resolve_executable(executable_path)
        self._verify_executable()

    def _resolve_executable(self, path: Optional[str] = None) -> str:
        """Resolve path to contrafold executable."""
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path

        # Check current directory
        local_path = os.path.join(os.getcwd(), "contrafold")
        if os.path.isfile(local_path) and os.access(local_path, os.X_OK):
            return local_path

        # Check PATH
        return "contrafold"  # Let the verification step handle this

    def _verify_executable(self) -> None:
        """Verify the executable exists and runs."""
        try:
            result = subprocess.run(
                [self.executable, "--help"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False
            )
            if result.returncode != 0:
                raise RuntimeError(f"ContraFold executable failed: {result.stderr.decode().strip()}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"ContraFold executable not found at '{self.executable}'. "
                "Please provide the correct path or ensure it's in your PATH."
            )

    def predict(self,
                seq_file: str,
                out_dir: Optional[str] = None,
                bpp: bool = True,
                gamma: Optional[float] = None,
                params_file: Optional[str] = None,
                constraints: Optional[str] = None) -> str:
        """
        Run ContraFold prediction on RNA sequence.

        Args:
            seq_file: Path to input RNA sequence file (FASTA format)
            out_dir: Directory for output files (created if not exists)
            bpp: Output base-pairing probabilities
            gamma: Tradeoff parameter for prediction
            params_file: Path to parameters file
            constraints: Path to constraints file

        Returns:
            Path to output file containing structure prediction
        """
        # Set up output directory
        output_dir = pathlib.Path(out_dir) if out_dir else pathlib.Path.cwd() / "contrafold_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "contrafold.out"

        # Construct command
        cmd = [self.executable, "predict", str(seq_file)]

        if bpp:
            cmd.append("--bpp")

        if gamma is not None:
            cmd.extend(["--gamma", str(gamma)])

        if params_file:
            cmd.extend(["--params", str(params_file)])

        if constraints:
            cmd.extend(["--constraints", str(constraints)])

        # Execute with proper output redirection
        with open(output_file, 'w') as f:
            try:
                result = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"ContraFold execution failed with code {e.returncode}: {e.stderr}")

        return str(output_file)

    def parse_output(self, output_file: str) -> Dict[str, Union[str, Dict[Tuple[int, int], float]]]:
        """
        Parse ContraFold output file.

        Args:
            output_file: Path to ContraFold output file

        Returns:
            Dictionary containing parsed structure and probabilities
        """
        result = {"structure": "", "pairs": {}}

        with open(output_file, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if line.startswith('>'):
                # Structure found in next line
                if i + 1 < len(lines):
                    result["structure"] = lines[i + 1].strip()
            elif line.strip() and ' ' in line:
                # Base pair probability line
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        i, j, prob = int(parts[0]), int(parts[1]), float(parts[2])
                        result["pairs"][(i, j)] = prob
                    except (ValueError, IndexError):
                        continue

        return result