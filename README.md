# AI to Params

A tool for converting AI-generated protein-ligand complexes to Rosetta params files, with optional relaxation and scoring via PyRosetta. This utility processes mmCIF files from AI modeling tools like AlphaFold3 and Chai, identifies ligand molecules, infers their bonds (using RDKit for proper bond orders), and generates Rosetta `.params` files for molecular dynamics simulations.

## Installation

### Prerequisites

- Python 3.8+
- BioPython for mmCIF file parsing
- RDKit for bond order inference

### Via `pip` (recommended)

```bash
git clone https://github.com/SiegelLab/general_protein_ml.git
cd general_protein_ml/ai_to_params
pip install -e .
```

This installs the `ai-to-params` console entry point with the `convert` subcommand.

### With relax/score support

Relaxation and scoring require PyRosetta, which is licensed through the University of Washington.

```bash
# First configure conda channel with your PyRosetta credentials:
# conda config --add channels https://{USERNAME}:{PASSWORD}@conda.graylab.jhu.edu
conda install pyrosetta
pip install -e ".[relax]"
```

### Direct script usage

If you prefer not to install the package, ensure these files stay together in the same directory:

- `ai_to_params.py` - Main conversion script
- `rosetta_params_utils.py` - Core parameterization utilities
- `constants.py` - Shared constants
- `cli.py` - Unified CLI entry point
- `relax_score.py` - PyRosetta relaxation and scoring

## Usage

The tool provides a unified CLI with four subcommands: `convert`, `relax`, `score`, and `run`.

### Convert (CIF to params)

```bash
ai-to-params convert -cif input.cif -prefix output_prefix
ai-to-params convert -cif input.cif -prefix output --clobber --clean-names
```

**Options:**

| Flag | Description |
|------|-------------|
| `-cif FILE` | Input mmCIF file (required) |
| `-prefix PREFIX` | Output prefix for generated files (required) |
| `--clobber` | Allow overwriting existing files |
| `--clean-names` | Use cleaned original names (ATP, GTP) instead of systematic names (L01, L02) |
| `--bond-tolerance FLOAT` | Bond inference tolerance in Angstroms (default: 0.45) |
| `--output-dir DIR` | Directory to write output files (default: `.`) |

### Relax (PyRosetta)

```bash
# Using ai_to_params output (auto-discovers PDB and params files)
ai-to-params relax --prefix my_complex --nstruct 5

# With explicit files
ai-to-params relax --pdb complex.pdb --params L01.params,L02.params

# With constraints
ai-to-params relax --prefix my_complex --constraints catalytic.cst --relax-mode torsional
```

**Options:**

| Flag | Description |
|------|-------------|
| `--prefix PREFIX` | Prefix for ai_to_params output (mutually exclusive with `--pdb`) |
| `--pdb FILE` | Path to input PDB file (mutually exclusive with `--prefix`) |
| `--params FILES` | Comma-separated .params files (required with `--pdb`) |
| `--nstruct N` | Number of structures to generate (default: 1) |
| `--relax-mode` | `cartesian` (default) or `torsional` |
| `--score-function` | Rosetta score function (default: `ref2015`) |
| `--constraints FILE` | Constraint file for relaxation |
| `--no-coord-constraints` | Disable coordinate constraints |
| `--output-dir DIR` | Output directory (default: `.`) |

### Score (PyRosetta, no relaxation)

Score a pose directly and compute interface energies without running FastRelax. Useful for quick evaluation of AI-predicted complexes.

```bash
# Using ai_to_params output
ai-to-params score --prefix my_complex

# With explicit files
ai-to-params score --pdb complex.pdb --params L01.params,L02.params
```

**Options:**

| Flag | Description |
|------|-------------|
| `--prefix PREFIX` | Prefix for ai_to_params output (mutually exclusive with `--pdb`) |
| `--pdb FILE` | Path to input PDB file (mutually exclusive with `--prefix`) |
| `--params FILES` | Comma-separated .params files (required with `--pdb`) |
| `--score-function` | Rosetta score function (default: `ref2015`) |
| `--output-dir DIR` | Directory to write summary CSV (default: `.`) |

Writes `{prefix}_score.csv` with total score and per-ligand interface energies.

### Full Pipeline (convert + relax)

```bash
ai-to-params run -cif input.cif -prefix output --nstruct 3
```

Accepts all options from both `convert` and `relax`.

### Examples

```bash
# Convert AlphaFold3 output
ai-to-params convert -cif AF3_output.cif -prefix my_complex

# Convert Chai output with original ligand names
ai-to-params convert -cif Chai_output.cif -prefix ligand_set --clean-names

# Full pipeline: convert, relax 5 structures, and score
ai-to-params run -cif AF3_output.cif -prefix my_complex --nstruct 5

# Relax with torsional mode and constraints
ai-to-params relax --prefix my_complex --relax-mode torsional --constraints design.cst

# Score a predicted complex without relaxation
ai-to-params score --prefix my_complex
```

## Output Files

### From `convert`
- `{prefix}_{ligand_name}.params` - Rosetta parameters file per ligand
- `{prefix}_{ligand_name}.pdb` - Individual ligand structure per ligand
- `{prefix}.pdb` - Full complex with sanitized names

### From `relax`
- `{prefix}_relaxed_NNNN.pdb` - Relaxed structures (one per nstruct)
- `{prefix}_summary.csv` - Scoring summary (total score, RMSD, interface energies)

### From `score`
- `{prefix}_score.csv` - Total score and per-ligand interface energies (no relaxation)

### File Naming Strategy
- **Filenames** use original ligand names from the CIF file (user-friendly)
- **Internal residue codes** use systematic names (L01, L02, etc.) for Rosetta consistency
- Use `--clean-names` to use cleaned original names instead of systematic names. Note: both Chai and AF3 are sometimes inconsistent with their naming, and Rosetta is somewhat strict about residue codes, so systematic names are recommended.

## How It Works

### 1. CIF Parsing
Parses mmCIF files using BioPython to extract atomic coordinates and residue information.

### 2. Ligand Identification
- Identifies HETATM records (non-standard residues)
- Filters out common solvents (HOH, WAT, SO4, PO4, CL)
- Groups unique ligands by residue name
- Handles metal ligands (MG, ZN, FE, CA, etc.) as special cases

### 3. Bond Inference
Uses RDKit's `DetermineBonds` for connectivity and bond order assignment (aromatic, double, triple bonds). Falls back to distance-based inference using covalent radii with a configurable tolerance (default: 0.45 A).

### 4. Rosetta Parameterization
Applies the complete Rosetta parameterization pipeline:
- Atom typing (Rosetta and molecular mechanics types)
- Partial charge assignment
- Bond analysis and rotatable bond identification
- Internal coordinate generation
- Fragment tree construction

### 5. Output Generation
Creates Rosetta-compatible `.params` files and cleaned PDB structures.

### 6. Relax and Score (optional)
- FastRelax with cartesian or torsional modes via PyRosetta
- Per-residue energy decomposition for interface energy calculation
- RMSD calculation relative to input structure
- CSV summary with scores across all generated structures

## Ligand Processing Rules

### Standard Ligands
- Small organic molecules are fully parameterized
- Generate both `.params` and `.pdb` files
- Bonds are inferred from 3D coordinates via RDKit

### Metal Ligands
- Metal residues (MG, ZN, FE, CA, etc.), including multi-atom metal clusters, are identified but not parameterized
- Skipped during file generation (metals don't need `.params` files in Rosetta)
- Still included in complex PDB with proper naming

### Ignored Residues
- Water molecules (HOH, WAT)
- Common ions and salts (SO4, PO4, CL)
- Standard amino acids and nucleotides

## Testing

```bash
# Run all tests with pytest
python -m pytest tests/ -v

# Specific test files
python -m pytest tests/test_bond_inference.py -v
python -m pytest tests/test_parameterization.py -v
```

### Available Tests
- `tests/test_atom_numbering.py` - Atom name sanitization
- `tests/test_metal_naming.py` - Metal atom handling
- `tests/test_chain_naming.py` - Chain ID sanitization
- `tests/test_fixes.py` - Various bug fixes
- `tests/test_constants.py` - Constants validation
- `tests/test_parameterization.py` - Parameterization pipeline
- `tests/test_bond_inference.py` - Bond inference and RDKit integration

## Troubleshooting

### Common Issues

**BioPython Import Error**
```
Error: BioPython is required but not installed.
```
Solution: `pip install biopython`

**RDKit Import Error**
```
Error: RDKit is required but not installed.
```
Solution: `pip install rdkit` or `conda install -c conda-forge rdkit`

**Missing Utility Functions**
```
Error importing required modules: No module named 'rosetta_params_utils'
```
Solution: Ensure all Python files are in the same directory, or install with `pip install -e .`

**File Overwrite Warning**
```
Warning: output.params exists, skipping (use --clobber to overwrite)
```
Solution: Use `--clobber` flag to allow overwriting existing files

**PyRosetta Not Found**
```
ImportError: No module named 'pyrosetta'
```
Solution: PyRosetta requires a license from UW. Install via `conda install pyrosetta` after configuring the conda channel.

### Bond Inference Issues
If bond inference produces unexpected results, adjust the tolerance with `--bond-tolerance` (default: 0.45 A).

## Technical Details

### Atom Name Sanitization
The tool handles various atom naming conventions from AI modeling outputs:
- `C_1` -> `C` (first occurrence gets base name)
- `C_2` -> `C1` (subsequent occurrences get numbered)
- `MG1_1` -> `MG` (metals use element name only)

### Chain ID Assignment
- Protein chains: Sanitized from original IDs (C_1 -> C, C_2 -> C1)
- Ligand chains: Systematic assignment (X, Y, Z, W, V, U, ...)

### Fragment Handling
Currently, all ligand atoms are assigned to a single fragment (fragment_id = 1). Multi-fragment ligands are not yet supported.

## License

MIT License - See LICENSE file for details.
