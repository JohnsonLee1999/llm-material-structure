# llm-material-structure
LLM-Driven Material Identification Framework

This project implements a Python-based framework that converts natural-language descriptions of materials into:

A structured, DFT-ready JSON representation
Input files for common DFT codes (VASP, Quantum ESPRESSO, CP2K)

The system combines:

Large Language Models (LLMs) for semantic understanding
Deterministic structure builders (pymatgen) for physically consistent crystal generation

## 1. Environment Setup

We recommend using conda.

### Create environment
conda create -n llm_material python=3.10

### Activate environment
conda activate llm_material

## 2. Installation

Install required dependencies:

pip install google-generativeai
pip install pymatgen
pip install python-dotenv

## 3. Gemini API Setup

This project uses the Google Gemini API.

### Step 1: Get API Key

Go to:
👉 https://aistudio.google.com

Create an API key.

### Step 2: Create .env file

In the project root directory:

touch .env

Add:

GEMINI_API_KEY=your_api_key_here

(Optional) you can also configure model:

GEMINI_MODEL=gemini-2.5-flash

## 4. How to Run
### Option 1: Run full pipeline
python main.py

Example input:

wurtzite gallium nitride

### Option 2: Run modules separately
**Step 1: Generate structure JSON**
python structure.py

**Step 2: Generate DFT input files**
python generate.py

## 5. Output
### Stage 1: Structure JSON

Example:

{
  "status": "fully_supported",
  "material_metadata": {
    "chemical_formula": "GaN",
    "structure_label": "wurtzite"
  },
  "lattice_parameters": {...},
  "atomic_positions": [...],
  "dft_computation_hints": {...}
}
### Stage 2: DFT Input Files

Depending on chosen software:

**VASP**
POSCAR
INCAR
KPOINTS
POTCAR (generated via pymatgen)
**Quantum ESPRESSO**
qe.in
**CP2K**
cp2k.inp

## 6. System Architecture

The framework follows a three-stage design:

User Input (Natural Language)
        ↓
LLM Semantic Parsing
        ↓
Deterministic Structure Builder (pymatgen)
        ↓
Structured JSON
        ↓
LLM-based Input File Generation
🔹 Key Design Choice

The LLM is used only for semantic interpretation,
while all crystal structures are generated deterministically.

This avoids:

hallucinated atomic coordinates
physically inconsistent structures
## 7. Structure Representation

We classify materials into two main categories:

### 3D Crystal Prototypes
fcc
bcc
diamond
hcp
rocksalt
zincblende
wurtzite
perovskite
rutile
anatase
### 2D Canonical Materials
graphene
monolayer h-BN
monolayer MoS₂
monolayer CrI₃

Each structure is generated using a deterministic builder.

## 8. Supported Modes
Status	Description
fully_supported	Full structure generated
candidate_only	Partial understanding, no structure generated
ambiguous	Input too vague

## 9. Limitations

The current framework does NOT support:

Defects and vacancies
Alloys and disorder
Surfaces and interfaces
Amorphous materials
Large supercells

## 10. Future Improvements
Template-based deterministic input generation
Support for defect structures
Better handling of magnetic configurations

## 11. Key Insight

This project demonstrates that:

LLMs are best used for interpretation,
not for directly generating physical structures.

By combining LLM reasoning with deterministic builders,
we achieve both flexibility and scientific reliability.
