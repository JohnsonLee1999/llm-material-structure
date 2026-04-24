import os
import json
import glob
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pymatgen.io.vasp.inputs import Potcar


# Optional proxy
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7890")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7890")


# ============================================================
# 1. Utilities
# ============================================================

def load_material_json(json_filename: str) -> Dict[str, Any]:
    if not os.path.exists(json_filename):
        raise FileNotFoundError(f"File not found: {json_filename}")

    with open(json_filename, "r", encoding="utf-8") as f:
        return json.load(f)


def choose_json_file_interactively() -> str:
    json_files = glob.glob("*.json")

    if len(json_files) == 0:
        raise FileNotFoundError("No JSON files found in the current directory. Please run structure.py first.")
    elif len(json_files) == 1:
        json_filename = json_files[0]
        print(f"🔍 Automatically detected the only JSON file: '{json_filename}'")
        return json_filename
    else:
        print("🔍 Multiple JSON files detected in the current directory:")
        for i, file in enumerate(json_files, 1):
            print(f"  {i}. {file}")

        choice = input("\nPlease enter the number or the exact filename to read:\n> ").strip()

        if choice.isdigit() and 1 <= int(choice) <= len(json_files):
            return json_files[int(choice) - 1]
        return choice


def choose_target_software_interactively() -> str:
    print("\n🛠️ Please select the target DFT software:")
    print("1. VASP")
    print("2. Quantum ESPRESSO")
    print("3. CP2K")

    software_choice = input("Enter the corresponding number or the software name directly:\n> ").strip().lower()

    if software_choice == "1" or ("vasp" in software_choice):
        return "VASP"
    elif software_choice == "2" or ("espresso" in software_choice) or ("qe" in software_choice):
        return "Quantum ESPRESSO"
    elif software_choice == "3" or ("cp2k" in software_choice):
        return "CP2K"
    else:
        return software_choice.upper()


def extract_unique_elements_from_atomic_positions(material_data: Dict[str, Any]) -> List[str]:
    atomic_positions = material_data.get("atomic_positions", [])
    ordered_elements = []
    for site in atomic_positions:
        el = site.get("element")
        if el and el not in ordered_elements:
            ordered_elements.append(el)
    return ordered_elements


def build_output_folder_name(material_data: Dict[str, Any], target_software: str) -> str:
    formula = material_data.get("material_metadata", {}).get("chemical_formula", "unknown")
    structure_label = material_data.get("material_metadata", {}).get("structure_label", "unknown")
    safe_formula = formula.replace(" ", "")
    safe_label = structure_label.replace(" ", "_")
    safe_software = target_software.replace(" ", "_")
    return f"{safe_formula}_{safe_label}_{safe_software}_inputs"


def validate_ready_for_generation(material_data: Dict[str, Any]) -> None:
    status = material_data.get("status", "fully_supported")
    if status != "fully_supported":
        message = material_data.get("validation", {}).get("message", "No detailed structure available.")
        raise ValueError(
            f"Cannot generate DFT inputs because structure status is '{status}'. {message}"
        )

    required_top_keys = ["material_metadata", "lattice_parameters", "atomic_positions", "dft_computation_hints"]
    for key in required_top_keys:
        if key not in material_data:
            raise ValueError(f"Input JSON is missing required key: '{key}'")

    if not material_data.get("atomic_positions"):
        raise ValueError("Input JSON contains no atomic positions.")


# ============================================================
# 2. Heuristics and prompting
# ============================================================

def get_software_heuristics(target_software: str) -> str:
    heuristics_registry = {
        "VASP": """
- Generate at minimum: POSCAR, INCAR, KPOINTS, and POTCAR_PLACEHOLDER.
- POSCAR must be in VASP 5 format.
- ENCUT:
  * If number of atoms < 50: use ENCUT = 520
  * Otherwise: use ENCUT = 400
- Geometry optimization:
  * IBRION = 2
  * NSW = 100
  * ISIF = 3 for bulk 3D systems
  * ISIF = 2 for 2D systems to avoid relaxing vacuum
- Electronic:
  * EDIFF = 1E-5
  * PREC = Accurate
- Spin:
  * If is_spin_polarized is true, set ISPIN = 2
  * If magnetic_state is ferromagnetic or antiferromagnetic, provide a reasonable MAGMOM line
- SOC:
  * If include_soc is true, include LSORBIT = .TRUE. and appropriate related settings
- DFT+U:
  * If hubbard_u_values is non-empty, include LDAU tags consistently
- KPOINTS:
  * Use Gamma-centered automatic mesh
  * Approximate with Length * k ≈ 40
  * If dimensionality is 2D, the vacuum direction must have exactly 1 k-point
- POTCAR:
  * Do NOT generate real POTCAR content
  * Instead output a key named POTCAR_PLACEHOLDER with a short note
""",
        "Quantum ESPRESSO": """
- Generate at minimum one main PWscf input file, e.g. qe.in.
- Use reasonable and consistent sections: &CONTROL, &SYSTEM, &ELECTRONS, optionally &IONS and &CELL.
- Geometry optimization:
  * Use calculation = 'vc-relax' for bulk 3D systems
  * Use calculation = 'relax' for 2D systems
- Cutoffs:
  * ecutwfc >= 60 Ry
  * ecutrho >= 240 Ry
- Occupations:
  * If metallic or uncertain, use smearing with smearing='mv' and degauss=0.02
  * Otherwise use fixed occupations where reasonable
- Spin:
  * If is_spin_polarized is true, enable nspin = 2 and provide starting_magnetization for relevant species
- SOC:
  * If include_soc is true, include noncolin=.true. and lspinorb=.true. where appropriate
- DFT+U:
  * If hubbard_u_values is non-empty, include Hubbard_U terms consistently
- K_POINTS:
  * Use automatic grid with Length * k ≈ 40
  * If dimensionality is 2D, the vacuum direction must have exactly 1 k-point
- Pseudopotentials:
  * In ATOMIC_SPECIES, use reasonable placeholder UPF names, e.g. Element.pbe-spn-kjpaw_psl.1.0.0.UPF
""",
        "CP2K": """
- Generate at minimum one main CP2K input file, e.g. cp2k.inp.
- Use a reasonable structure with sections such as:
  * &GLOBAL
  * &FORCE_EVAL
  * &DFT
  * &SUBSYS
  * &MOTION if geometry optimization is needed
- Run type:
  * GEO_OPT for geometry optimization
  * ENERGY_FORCE for static single-point calculations
- Basis/potential:
  * Assign BASIS_SET DZVP-MOLOPT-SR-GTH
  * Assign POTENTIAL GTH-PBE
- Grid:
  * Use CUTOFF >= 500
  * REL_CUTOFF = 60
- Spin:
  * If is_spin_polarized is true, include UKS .TRUE.
  * If magnetic_state suggests magnetism, set multiplicity reasonably
- SOC:
  * If include_soc is true and implementation would be ambiguous, mention it in a comment rather than inventing unsupported syntax
- K-points:
  * If k-points are used, ensure the vacuum direction has 1 for 2D materials
- Coordinates:
  * Use the coordinates from the input JSON faithfully
"""
    }

    return heuristics_registry.get(
        target_software,
        "- Apply standard best-practice settings for this DFT code."
    )


def build_generation_prompt(material_data: Dict[str, Any], target_software: str) -> str:
    specific_heuristics = get_software_heuristics(target_software)

    structure_label = material_data.get("material_metadata", {}).get("structure_label", "unknown")
    structure_family = material_data.get("material_metadata", {}).get("structure_family", "unknown")
    formula = material_data.get("material_metadata", {}).get("chemical_formula", "unknown")
    dimensionality = material_data.get("dft_computation_hints", {}).get("dimensionality", "unknown")

    return f"""
You are an expert computational materials scientist specializing in {target_software}.

I will provide you with a JSON object containing a material structure and calculation hints.
The structure is already validated and should be translated faithfully into {target_software} input files.

Material summary:
- Formula: {formula}
- Structure family: {structure_family}
- Structure label: {structure_label}
- Dimensionality: {dimensionality}

CRITICAL INSTRUCTIONS:
1. Use the lattice parameters and atomic fractional coordinates exactly as given.
2. Respect all calculation hints, including spin polarization, magnetism, SOC, Hubbard U, and dimensionality.
3. Generate input files suitable for a high-quality starting DFT calculation.
4. Do not invent new atoms, remove atoms, or change the structure topology.
5. If something is uncertain, prefer a conservative, standard setup.
6. Output MUST be a raw JSON object where:
   - keys = filenames
   - values = full plain-text file contents
7. Do not include markdown fences like ```json.
8. Keep comments inside generated files minimal and professional.

SOFTWARE-SPECIFIC HEURISTICS:
{specific_heuristics}
"""


# ============================================================
# 3. LLM generation
# ============================================================

def call_gemini_generate_files(material_data: Dict[str, Any], target_software: str) -> Dict[str, str]:
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY not found. Please check your .env file.")

    client = genai.Client()
    system_instruction = build_generation_prompt(material_data, target_software)

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=json.dumps(material_data, indent=2),
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            temperature=0.2,
        )
    )

    try:
        generated_files = json.loads(response.text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Gemini did not return valid JSON. Raw response:\n{response.text}") from e

    if not isinstance(generated_files, dict):
        raise ValueError("Generated result is not a JSON object mapping filenames to file contents.")

    for filename, content in generated_files.items():
        if not isinstance(filename, str) or not isinstance(content, str):
            raise ValueError("Generated JSON must map string filenames to string file contents.")

    return generated_files


# ============================================================
# 4. File writing
# ============================================================

def write_generated_files(folder_name: str, generated_files: Dict[str, str]) -> None:
    os.makedirs(folder_name, exist_ok=True)

    print(f"\n✅ Writing generated files into directory: '{folder_name}'")
    for filename, file_content in generated_files.items():
        if "POTCAR" in filename:
            # POTCAR placeholder is intentionally skipped as a real file here;
            # real POTCAR generation is handled separately for VASP.
            continue

        filepath = os.path.join(folder_name, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(file_content)
        print(f"  - {filename}")


def try_generate_potcar(folder_name: str, material_data: Dict[str, Any], generated_files: Dict[str, str]) -> None:
    """
    Assemble a real POTCAR using pymatgen if possible.
    Priority:
    1. Extract element order from POSCAR line 6 (VASP5 format)
    2. Fallback to atomic_positions unique element order
    """
    poscar_content = generated_files.get("POSCAR")
    elements_line = None

    if poscar_content:
        lines = poscar_content.strip().splitlines()
        if len(lines) > 5:
            candidate = lines[5].split()
            if candidate and all(token.isalpha() for token in candidate):
                elements_line = candidate

    if not elements_line:
        elements_line = extract_unique_elements_from_atomic_positions(material_data)

    if not elements_line:
        print("  ⚠️ Could not determine element order for POTCAR generation.")
        return

    print(f"  ⚙️ Calling pymatgen to assemble POTCAR for {elements_line} ...")
    try:
        potcar = Potcar(elements_line, functional="PBE")
        potcar.write_file(os.path.join(folder_name, "POTCAR"))
        print("  + POTCAR (PBE) successfully generated (powered by pymatgen).")
    except Exception as e:
        print(f"  ⚠️ pymatgen failed to generate POTCAR. Ensure you have configured VASP pseudopotential paths via 'pmg config'. Error: {e}")


# ============================================================
# 5. Main entry
# ============================================================

def run_generate(json_filename: Optional[str] = None, target_software: Optional[str] = None) -> None:
    """
    Reads a JSON structure file and generates DFT input files for the chosen software.
    """
    if not json_filename:
        json_filename = choose_json_file_interactively()

    material_data = load_material_json(json_filename)

    try:
        validate_ready_for_generation(material_data)
    except ValueError as e:
        print(f"\n❌ {e}")
        return

    if not target_software:
        target_software = choose_target_software_interactively()

    print(f"\n🚀 Requesting Gemini to generate input files for {target_software} ...")

    try:
        generated_files = call_gemini_generate_files(material_data, target_software)
    except Exception as e:
        print(f"\n❌ Failed to generate input files: {e}")
        return

    folder_name = build_output_folder_name(material_data, target_software)
    write_generated_files(folder_name, generated_files)

    if target_software == "VASP":
        try_generate_potcar(folder_name, material_data, generated_files)

    print(f"\nYou can check the generated input files in the '{folder_name}' directory.")


if __name__ == "__main__":
    run_generate()