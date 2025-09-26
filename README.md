# AI to Params

A tool for converting AI-generated protein-ligand complexes to Rosetta params files. This utility processes mmCIF files from AI modeling tools like AlphaFold3 and Chai, identifies ligand molecules, infers their bonds, and generates Rosetta `.params` files for molecular dynamics simulations.

## Installation

### Prerequisites

- Python 3.7 or higher
- BioPython for mmCIF file parsing

### Install Dependencies

```bash
pip install biopython
```

### Setup

1. Clone or download this repository
2. Ensure all Python files are in the same directory:
   - `ai_to_params.py` - Main conversion script
   - `rosetta_params_utils.py` - Core parameterization utilities

## Usage

### Basic Usage

```bash
python ai_to_params.py -cif input.cif -prefix output_prefix
```

### Command Line Options

- `-cif FILE` - Path to input mmCIF file (required)
- `-prefix PREFIX` - Output prefix for all generated files (required)
- `--clobber` - Allow overwriting existing files
- `--clean-names` - Use cleaned original names (ATP, GTP) instead of systematic names (L01, L02)

### Examples

```bash
# Process AlphaFold3 output with systematic naming
python ai_to_params.py -cif AF3_output.cif -prefix my_complex

# Process Chai output with original ligand names
python ai_to_params.py -cif Chai_output.cif -prefix ligand_set --clean-names

# Allow overwriting existing output files
python ai_to_params.py -cif input.cif -prefix test --clobber
```

## Output Files

The tool generates several output files for each ligand found:

### Per-Ligand Files
- `{prefix}_{ligand_name}.params` - Rosetta parameters file
- `{prefix}_{ligand_name}.pdb` - Individual ligand structure

### Complex File
- `{prefix}_cleaned.pdb` - Cleaned structure of entire complex with sanitized names

### File Naming Strategy
- **Filenames** use original ligand names from the CIF file (user-friendly)
- **Internal residue codes** use systematic names (L01, L02, etc.) for Rosetta consistency
- Use `--clean-names` to use cleaned original names instead of systematic names

## How It Works

### 1. CIF Parsing
Parses mmCIF files using BioPython to extract atomic coordinates and residue information.

### 2. Ligand Identification
- Identifies HETATM records (non-standard residues)
- Filters out common solvents (HOH, WAT, SO4, PO4, CL)
- Groups unique ligands by residue name
- Handles metal ligands (MG, ZN, FE, CA, etc.) as special cases

### 3. Bond Inference
Automatically infers chemical bonds based on:
- Covalent radii of atoms
- Inter-atomic distances
- Configurable tolerance (default: 0.45 Å)

### 4. Rosetta Parameterization
Applies the complete Rosetta parameterization pipeline:
- Atom typing (Rosetta and molecular mechanics types)
- Partial charge assignment
- Bond analysis and rotatable bond identification
- Internal coordinate generation
- Fragment tree construction

### 5. Output Generation
Creates Rosetta-compatible `.params` files and cleaned PDB structures.

## Ligand Processing Rules

### Standard Ligands
- Small organic molecules are fully parameterized
- Generate both `.params` and `.pdb` files
- Bonds are inferred from 3D coordinates

### Metal Ligands
- Single-atom metals (MG, ZN, FE, etc.) are identified but not parameterized
- Skipped during file generation (metals don't need `.params` files in Rosetta)
- Still included in cleaned complex PDB with proper naming

### Ignored Residues
- Water molecules (HOH, WAT)
- Common ions and salts (SO4, PO4, CL)
- Standard amino acids and nucleotides

## Testing

The repository includes test scripts to verify functionality:

### Available Tests
- `tests/test_atom_numbering.py` - Tests atom name sanitization
- `tests/test_metal_naming.py` - Tests metal atom handling
- `tests/test_fixes.py` - Tests various bug fixes
- `tests/test_chain_naming.py` - Tests chain ID sanitization

### Running Tests

```bash
# Run individual tests
python tests/test_atom_numbering.py
python tests/test_metal_naming.py

# Run all tests
python tests/test_*.py
```

Test output shows which functions pass (✓) or fail (✗) with expected vs actual results.

## Troubleshooting

### Common Issues

**BioPython Import Error**
```
Error: BioPython is required but not installed.
```
Solution: Install BioPython with `pip install biopython`

**Missing Utility Functions**
```
Error importing required modules: No module named 'rosetta_params_utils'
```
Solution: Ensure `rosetta_params_utils.py` is in the same directory as `ai_to_params.py`

**File Overwrite Warning**
```
Warning: output.params exists, skipping (use --clobber to overwrite)
```
Solution: Use `--clobber` flag to allow overwriting existing files

### Bond Inference Issues
If bond inference produces unexpected results, the tolerance can be adjusted by modifying the `tolerance` parameter in the `infer_bonds()` function (default: 0.45 Å).

## Technical Details

### Atom Name Sanitization
The tool handles various atom naming conventions from AI modeling outputs:
- `C_1` → `C` (first occurrence gets base name)
- `C_2` → `C1` (subsequent occurrences get numbered)
- `MG1_1` → `MG` (metals use element name only)

### Chain ID Assignment
- Protein chains: Sanitized from original IDs (C_1 → C, C_2 → C1)
- Ligand chains: Systematic assignment (X, Y, Z, W, V, U, ...)

### Fragment Handling
Currently, all ligand atoms are assigned to a single fragment (fragment_id = 1). Multi-fragment ligands are not yet supported.

## License

MIT License - See LICENSE file for details.