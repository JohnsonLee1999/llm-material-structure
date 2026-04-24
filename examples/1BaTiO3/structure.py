import os
import re
import json
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai import types

from pymatgen.core import Lattice, Structure, Element, Composition


# Optional proxy
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7890")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7890")


# ============================================================
# 1. Metadata and default parameters
# ============================================================

STRUCTURE_LIBRARY: Dict[str, Dict[str, str]] = {
    # 3D prototypes
    "fcc": {"family": "3D_prototype", "space_group": "Fm-3m", "crystal_system": "cubic"},
    "bcc": {"family": "3D_prototype", "space_group": "Im-3m", "crystal_system": "cubic"},
    "diamond": {"family": "3D_prototype", "space_group": "Fd-3m", "crystal_system": "cubic"},
    "hcp": {"family": "3D_prototype", "space_group": "P6_3/mmc", "crystal_system": "hexagonal"},
    "rocksalt": {"family": "3D_prototype", "space_group": "Fm-3m", "crystal_system": "cubic"},
    "zincblende": {"family": "3D_prototype", "space_group": "F-43m", "crystal_system": "cubic"},
    "wurtzite": {"family": "3D_prototype", "space_group": "P6_3mc", "crystal_system": "hexagonal"},
    "perovskite": {"family": "3D_prototype", "space_group": "Pm-3m", "crystal_system": "cubic"},
    "rutile": {"family": "3D_prototype", "space_group": "P4_2/mnm", "crystal_system": "tetragonal"},
    "anatase": {"family": "3D_prototype", "space_group": "I4_1/amd", "crystal_system": "tetragonal"},

    # 2D canonical materials
    "graphene": {"family": "2D_canonical_material", "space_group": "P6/mmm", "crystal_system": "hexagonal"},
    "monolayer_hbn": {"family": "2D_canonical_material", "space_group": "P6/mmm", "crystal_system": "hexagonal"},
    "monolayer_mos2": {"family": "2D_canonical_material", "space_group": "P-6m2", "crystal_system": "hexagonal"},
    "monolayer_cri3": {"family": "2D_canonical_material", "space_group": "P-31m", "crystal_system": "trigonal"},
}


ELEMENT_NAME_MAP = {
    "silicon": "Si",
    "copper": "Cu",
    "gallium": "Ga",
    "nitrogen": "N",
    "barium": "Ba",
    "titanium": "Ti",
    "oxygen": "O",
    "carbon": "C",
    "boron": "B",
    "sodium": "Na",
    "chlorine": "Cl",
    "zinc": "Zn",
    "sulfur": "S",
    "molybdenum": "Mo",
    "iodine": "I",
    "chromium": "Cr",
    "aluminum": "Al",
    "nickel": "Ni",
    "iron": "Fe",
    "magnesium": "Mg",
    "calcium": "Ca",
    "strontium": "Sr",
}


ELEMENTAL_LATTICE_A = {
    "Cu": 3.615,
    "Al": 4.05,
    "Ni": 3.52,
    "Fe": 2.87,
    "Si": 5.43,
    "C": 3.567,
    "Mg": 3.21,
}

BINARY_DEFAULTS = {
    ("Ga", "N", "wurtzite"): {"a": 3.189, "c": 5.185, "u": 0.377},
    ("Ga", "N", "zincblende"): {"a": 4.50},
    ("Zn", "S", "zincblende"): {"a": 5.41},
    ("Na", "Cl", "rocksalt"): {"a": 5.64},
    ("Mg", "O", "rocksalt"): {"a": 4.21},
    ("Ti", "O", "rutile"): {"a": 4.594, "c": 2.959, "u": 0.305},
    ("Ti", "O", "anatase"): {"a": 3.784, "c": 9.515, "u": 0.208},
    ("B", "N", "monolayer_hbn"): {"a": 2.50},
    ("Mo", "S", "monolayer_mos2"): {"a": 3.18},
}

TERNARY_DEFAULTS = {
    ("Ba", "Ti", "O", "perovskite"): {"a": 4.01},
    ("Sr", "Ti", "O", "perovskite"): {"a": 3.905},
    ("Ca", "Ti", "O", "perovskite"): {"a": 3.80},
}


# ============================================================
# 2. LLM semantic parser
# ============================================================

def call_gemini_parser(user_input: str) -> Dict[str, Any]:
    """
    LLM is used only for semantic interpretation.
    It does NOT generate atom coordinates.
    """
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY not found. Please check your .env file.")

    client = genai.Client()

    system_instruction = """
You are an expert computational materials scientist.

Your task is ONLY to parse the user's natural-language material description into a normalized JSON intent.
Do NOT generate atomic coordinates.
Do NOT invent detailed crystallographic values unless explicitly specified by the user.

Return ONLY a raw JSON object with exactly this schema:
{
  "chemical_formula": "string or null",
  "elements": ["element symbols"],
  "structure_label": "fcc | bcc | diamond | hcp | rocksalt | zincblende | wurtzite | perovskite | rutile | anatase | graphene | monolayer_hbn | monolayer_mos2 | monolayer_cri3 | unknown",
  "structure_family": "3D_prototype | 2D_canonical_material | unknown",
  "dimensionality": "bulk | 2D | 1D | 0D | unknown",
  "magnetic_state": "non-magnetic | ferromagnetic | antiferromagnetic | unknown",
  "is_spin_polarized": false,
  "include_soc": false,
  "calculation_task": "geometry_optimization | static_scf | default",
  "user_lattice_hints": {
    "a": null,
    "b": null,
    "c": null,
    "alpha": null,
    "beta": null,
    "gamma": null
  },
  "confidence": 0.0,
  "notes": "brief explanation"
}

Rules:
- "face-centered cubic" or "face-centred cubic" -> structure_label = "fcc"
- "body-centered cubic" or "body-centred cubic" -> structure_label = "bcc"
- "diamond cubic silicon" -> formula = "Si", structure_label = "diamond"
- "wurtzite gallium nitride" -> formula = "GaN", structure_label = "wurtzite"
- "perovskite oxide with barium and titanium" -> formula = "BaTiO3", structure_label = "perovskite"
- "rutile titanium dioxide" -> formula = "TiO2", structure_label = "rutile"
- "anatase titanium dioxide" -> formula = "TiO2", structure_label = "anatase"
- "graphene" -> formula = "C", structure_label = "graphene", structure_family = "2D_canonical_material", dimensionality = "2D"
- "hexagonal boron nitride" or "monolayer h-BN" -> formula = "BN", structure_label = "monolayer_hbn" if monolayer/single-layer is mentioned; otherwise use "unknown" if ambiguous
- "monolayer MoS2" -> formula = "MoS2", structure_label = "monolayer_mos2", structure_family = "2D_canonical_material", dimensionality = "2D"
- "monolayer CrI3 with FM order" -> formula = "CrI3", structure_label = "monolayer_cri3", structure_family = "2D_canonical_material", dimensionality = "2D", magnetic_state = "ferromagnetic", is_spin_polarized = true
- If the material is magnetic or the user specifies FM/AFM, set is_spin_polarized = true
- If uncertain, use unknown/null instead of inventing
Return only raw JSON.
"""

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=user_input,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            temperature=0.1,
        ),
    )

    return json.loads(response.text)


# ============================================================
# 3. Rule-based fallback parser
# ============================================================

def extract_elements_from_text(text: str) -> List[str]:
    found = []

    formula_tokens = re.findall(r"\b([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)+)\b", text)
    for token in formula_tokens:
        try:
            comp = Composition(token)
            for el in comp.as_dict().keys():
                if el not in found:
                    found.append(el)
        except Exception:
            pass

    lower_text = text.lower()
    for name, sym in ELEMENT_NAME_MAP.items():
        if name in lower_text and sym not in found:
            found.append(sym)

    symbol_matches = re.findall(r"\b([A-Z][a-z]?)\b", text)
    for sym in symbol_matches:
        if Element.is_valid_symbol(sym) and sym not in found:
            found.append(sym)

    return found


def infer_formula(text: str, elements: List[str], structure_label: str) -> Optional[str]:
    lower_text = text.lower()

    explicit_formula = re.findall(r"\b([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)+)\b", text)
    if explicit_formula:
        return explicit_formula[0]

    if "gallium nitride" in lower_text:
        return "GaN"
    if "silicon" in lower_text:
        return "Si"
    if "copper" in lower_text:
        return "Cu"
    if "graphene" in lower_text:
        return "C"
    if "hexagonal boron nitride" in lower_text or "h-bn" in lower_text or "hbn" in lower_text:
        return "BN"
    if "mos2" in lower_text or ("molybdenum" in lower_text and "sulfur" in lower_text):
        return "MoS2"
    if "cri3" in lower_text or ("chromium" in lower_text and "iodine" in lower_text):
        return "CrI3"
    if "tio2" in lower_text or ("titanium" in lower_text and "oxygen" in lower_text and structure_label in {"rutile", "anatase"}):
        return "TiO2"
    if "barium" in lower_text and "titanium" in lower_text and "perovskite" in lower_text:
        return "BaTiO3"
    if "strontium" in lower_text and "titanium" in lower_text and "perovskite" in lower_text:
        return "SrTiO3"

    if len(elements) == 1:
        return elements[0]

    return None


def fallback_parse(user_input: str) -> Dict[str, Any]:
    text = user_input.strip().lower()

    structure_label = "unknown"
    structure_family = "unknown"
    dimensionality = "bulk"
    magnetic_state = "unknown"
    is_spin_polarized = False
    include_soc = False
    calculation_task = "geometry_optimization"
    confidence = 0.50

    if "face-centred cubic" in text or "face-centered cubic" in text or re.search(r"\bfcc\b", text):
        structure_label = "fcc"
        structure_family = "3D_prototype"
        confidence = 0.95
    elif re.search(r"\bbcc\b", text) or "body-centred cubic" in text or "body-centered cubic" in text:
        structure_label = "bcc"
        structure_family = "3D_prototype"
        confidence = 0.95
    elif "diamond cubic" in text or ("diamond" in text and "silicon" in text):
        structure_label = "diamond"
        structure_family = "3D_prototype"
        confidence = 0.95
    elif "hcp" in text or "hexagonal close packed" in text:
        structure_label = "hcp"
        structure_family = "3D_prototype"
        confidence = 0.90
    elif "rocksalt" in text or "rock salt" in text:
        structure_label = "rocksalt"
        structure_family = "3D_prototype"
        confidence = 0.90
    elif "zincblende" in text or "zinc blende" in text:
        structure_label = "zincblende"
        structure_family = "3D_prototype"
        confidence = 0.90
    elif "wurtzite" in text:
        structure_label = "wurtzite"
        structure_family = "3D_prototype"
        confidence = 0.95
    elif "perovskite" in text:
        structure_label = "perovskite"
        structure_family = "3D_prototype"
        confidence = 0.85
    elif "rutile" in text:
        structure_label = "rutile"
        structure_family = "3D_prototype"
        confidence = 0.90
    elif "anatase" in text:
        structure_label = "anatase"
        structure_family = "3D_prototype"
        confidence = 0.90
    elif "graphene" in text:
        structure_label = "graphene"
        structure_family = "2D_canonical_material"
        dimensionality = "2D"
        confidence = 0.98
    elif ("hexagonal boron nitride" in text or "h-bn" in text or "hbn" in text) and ("monolayer" in text or "single layer" in text or "2d" in text):
        structure_label = "monolayer_hbn"
        structure_family = "2D_canonical_material"
        dimensionality = "2D"
        confidence = 0.95
    elif ("mos2" in text or ("molybdenum" in text and "sulfur" in text)) and ("monolayer" in text or "single layer" in text or "2d" in text):
        structure_label = "monolayer_mos2"
        structure_family = "2D_canonical_material"
        dimensionality = "2D"
        confidence = 0.95
    elif ("cri3" in text or ("chromium" in text and "iodine" in text)) and ("monolayer" in text or "single layer" in text or "2d" in text):
        structure_label = "monolayer_cri3"
        structure_family = "2D_canonical_material"
        dimensionality = "2D"
        confidence = 0.96

    if "fm" in text or "ferromagnetic" in text:
        magnetic_state = "ferromagnetic"
        is_spin_polarized = True
    elif "afm" in text or "antiferromagnetic" in text:
        magnetic_state = "antiferromagnetic"
        is_spin_polarized = True
    elif "non-magnetic" in text or "nonmagnetic" in text:
        magnetic_state = "non-magnetic"
        is_spin_polarized = False

    if structure_label == "monolayer_cri3" and magnetic_state == "unknown":
        magnetic_state = "ferromagnetic"
        is_spin_polarized = True

    if "soc" in text or "spin orbit" in text or "spin-orbit" in text:
        include_soc = True

    elements = extract_elements_from_text(user_input)
    chemical_formula = infer_formula(user_input, elements, structure_label)

    return {
        "chemical_formula": chemical_formula,
        "elements": elements,
        "structure_label": structure_label,
        "structure_family": structure_family,
        "dimensionality": dimensionality,
        "magnetic_state": magnetic_state,
        "is_spin_polarized": is_spin_polarized,
        "include_soc": include_soc,
        "calculation_task": calculation_task,
        "user_lattice_hints": {
            "a": None, "b": None, "c": None,
            "alpha": None, "beta": None, "gamma": None
        },
        "confidence": confidence,
        "notes": "Fallback rule-based parser used."
    }


# ============================================================
# 4. Builder helpers
# ============================================================

def get_binary_default(elements: List[str], key: str) -> Dict[str, Any]:
    a, b = elements[0], elements[1]
    return BINARY_DEFAULTS.get((a, b, key), {}) or BINARY_DEFAULTS.get((b, a, key), {})


def get_ternary_default(elements: List[str], key: str) -> Dict[str, Any]:
    if len(elements) != 3 or "O" not in elements:
        return {}
    others = [e for e in elements if e != "O"]
    a, b = others[0], others[1]
    return TERNARY_DEFAULTS.get((a, b, "O", key), {}) or TERNARY_DEFAULTS.get((b, a, "O", key), {})


def add_vacuum_to_structure(structure: Structure, vacuum_z: float) -> Structure:
    old = structure.lattice
    new_c = old.c + vacuum_z
    new_lattice = Lattice.from_parameters(old.a, old.b, new_c, old.alpha, old.beta, old.gamma)

    cart = structure.cart_coords.copy()
    zmin = cart[:, 2].min()
    zmax = cart[:, 2].max()
    slab_center = 0.5 * (zmin + zmax)
    target_center = new_c / 2.0
    shift = target_center - slab_center
    cart[:, 2] += shift

    return Structure(new_lattice, structure.species, cart, coords_are_cartesian=True)


# ============================================================
# 5. 3D prototype builders
# ============================================================

def build_fcc(formula: str, hints: Dict[str, Any]) -> Structure:
    elems = list(Composition(formula).get_el_amt_dict().keys())
    if len(elems) != 1:
        raise ValueError("FCC builder currently supports elemental FCC only.")
    el = elems[0]
    a = hints.get("a") or ELEMENTAL_LATTICE_A.get(el, 3.60)
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(a), [el], [[0, 0, 0]])


def build_bcc(formula: str, hints: Dict[str, Any]) -> Structure:
    elems = list(Composition(formula).get_el_amt_dict().keys())
    if len(elems) != 1:
        raise ValueError("BCC builder currently supports elemental BCC only.")
    el = elems[0]
    a = hints.get("a") or ELEMENTAL_LATTICE_A.get(el, 2.87)
    return Structure.from_spacegroup("Im-3m", Lattice.cubic(a), [el], [[0, 0, 0]])


def build_diamond(formula: str, hints: Dict[str, Any]) -> Structure:
    elems = list(Composition(formula).get_el_amt_dict().keys())
    if len(elems) != 1:
        raise ValueError("Diamond builder currently supports elemental diamond structures only.")
    el = elems[0]
    a = hints.get("a") or ELEMENTAL_LATTICE_A.get(el, 5.43)
    return Structure.from_spacegroup("Fd-3m", Lattice.cubic(a), [el], [[0, 0, 0]])


def build_hcp(formula: str, hints: Dict[str, Any]) -> Structure:
    elems = list(Composition(formula).get_el_amt_dict().keys())
    if len(elems) != 1:
        raise ValueError("HCP builder currently supports elemental HCP only.")
    el = elems[0]
    a = hints.get("a") or ELEMENTAL_LATTICE_A.get(el, 3.21)
    c = hints.get("c") or (1.633 * a)
    lattice = Lattice.hexagonal(a, c)
    return Structure.from_spacegroup("P6_3/mmc", lattice, [el], [[1/3, 2/3, 1/4]])


def build_rocksalt(elements: List[str], hints: Dict[str, Any]) -> Structure:
    if len(elements) != 2:
        raise ValueError("Rocksalt builder requires a binary formula.")
    defaults = get_binary_default(elements, "rocksalt")
    a = hints.get("a") or defaults.get("a") or 5.5
    lattice = Lattice.cubic(a)
    return Structure.from_spacegroup(
        "Fm-3m",
        lattice,
        [elements[0], elements[1]],
        [[0, 0, 0], [0.5, 0.5, 0.5]]
    )


def build_zincblende(elements: List[str], hints: Dict[str, Any]) -> Structure:
    if len(elements) != 2:
        raise ValueError("Zincblende builder requires a binary formula.")
    defaults = get_binary_default(elements, "zincblende")
    a = hints.get("a") or defaults.get("a") or 5.45
    lattice = Lattice.cubic(a)
    return Structure.from_spacegroup(
        "F-43m",
        lattice,
        [elements[0], elements[1]],
        [[0, 0, 0], [0.25, 0.25, 0.25]]
    )


def build_wurtzite(elements: List[str], hints: Dict[str, Any]) -> Structure:
    if len(elements) != 2:
        raise ValueError("Wurtzite builder requires a binary formula.")
    defaults = get_binary_default(elements, "wurtzite")
    a = hints.get("a") or defaults.get("a") or 3.2
    c = hints.get("c") or defaults.get("c") or 5.2
    u = defaults.get("u", 0.375)

    lattice = Lattice.hexagonal(a, c)
    species = [elements[0], elements[0], elements[1], elements[1]]
    frac_coords = [
        [1/3, 2/3, 0.0],
        [2/3, 1/3, 0.5],
        [1/3, 2/3, u],
        [2/3, 1/3, (0.5 + u) % 1.0],
    ]
    return Structure(lattice, species, frac_coords)


def build_perovskite(elements: List[str], hints: Dict[str, Any]) -> Structure:
    if len(elements) != 3 or "O" not in elements:
        raise ValueError("Perovskite builder currently supports oxide ABO3 only.")
    defaults = get_ternary_default(elements, "perovskite")
    a = hints.get("a") or defaults.get("a") or 4.0

    others = [e for e in elements if e != "O"]
    A, B = others[0], others[1]
    lattice = Lattice.cubic(a)
    return Structure.from_spacegroup(
        "Pm-3m",
        lattice,
        [A, B, "O"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.0]]
    )


def build_rutile(elements: List[str], hints: Dict[str, Any]) -> Structure:
    if len(elements) != 2 or "O" not in elements:
        raise ValueError("Rutile builder currently supports binary oxides like TiO2.")
    metal = [e for e in elements if e != "O"][0]
    defaults = get_binary_default([metal, "O"], "rutile")
    a = hints.get("a") or defaults.get("a") or 4.594
    c = hints.get("c") or defaults.get("c") or 2.959
    u = defaults.get("u", 0.305)

    lattice = Lattice.tetragonal(a, c)
    return Structure.from_spacegroup(
        "P4_2/mnm",
        lattice,
        [metal, "O"],
        [[0, 0, 0], [u, u, 0]]
    )


def build_anatase(elements: List[str], hints: Dict[str, Any]) -> Structure:
    if len(elements) != 2 or "O" not in elements:
        raise ValueError("Anatase builder currently supports binary oxides like TiO2.")
    metal = [e for e in elements if e != "O"][0]
    defaults = get_binary_default([metal, "O"], "anatase")
    a = hints.get("a") or defaults.get("a") or 3.784
    c = hints.get("c") or defaults.get("c") or 9.515
    u = defaults.get("u", 0.208)

    lattice = Lattice.tetragonal(a, c)
    return Structure.from_spacegroup(
        "I4_1/amd",
        lattice,
        [metal, "O"],
        [[0, 0, 0], [0, 0, u]]
    )


# ============================================================
# 6. 2D canonical material builders
# ============================================================

def build_graphene(hints: Dict[str, Any]) -> Structure:
    a = hints.get("a") or 2.46
    c = hints.get("c") or 3.35
    lattice = Lattice.hexagonal(a, c)
    species = ["C", "C"]
    coords = [
        [0.0, 0.0, 0.5],
        [1/3, 2/3, 0.5],
    ]
    return Structure(lattice, species, coords)


def build_monolayer_hbn(hints: Dict[str, Any]) -> Structure:
    a = hints.get("a") or get_binary_default(["B", "N"], "monolayer_hbn").get("a", 2.50)
    c = hints.get("c") or 3.33
    lattice = Lattice.hexagonal(a, c)
    species = ["B", "N"]
    coords = [
        [0.0, 0.0, 0.5],
        [1/3, 2/3, 0.5],
    ]
    return Structure(lattice, species, coords)


def build_monolayer_mos2(hints: Dict[str, Any]) -> Structure:
    a = hints.get("a") or get_binary_default(["Mo", "S"], "monolayer_mos2").get("a", 3.18)
    c = hints.get("c") or 6.50
    z = 0.12
    lattice = Lattice.hexagonal(a, c)
    species = ["Mo", "S", "S"]
    coords = [
        [0.0, 0.0, 0.5],
        [1/3, 2/3, 0.5 + z],
        [1/3, 2/3, 0.5 - z],
    ]
    return Structure(lattice, species, coords)


def build_monolayer_cri3(hints: Dict[str, Any]) -> Structure:
    a = hints.get("a") or 6.87
    c = hints.get("c") or 7.00
    lattice = Lattice.hexagonal(a, c)
    species = ["Cr", "Cr", "I", "I", "I", "I", "I", "I"]
    coords = [
        [0.0, 0.0, 0.5],
        [2/3, 1/3, 0.5],
        [1/3, 2/3, 0.35],
        [1/3, 2/3, 0.65],
        [0.0, 0.0, 0.35],
        [0.0, 0.0, 0.65],
        [2/3, 1/3, 0.35],
        [2/3, 1/3, 0.65],
    ]
    return Structure(lattice, species, coords)


# ============================================================
# 7. Deterministic builder router
# ============================================================

def build_supported_structure(parsed: Dict[str, Any]) -> Tuple[Structure, Dict[str, Any]]:
    structure_label = parsed.get("structure_label", "unknown")
    structure_family = parsed.get("structure_family", "unknown")
    chemical_formula = parsed.get("chemical_formula")
    hints = parsed.get("user_lattice_hints", {}) or {}
    dimensionality = parsed.get("dimensionality", "bulk")

    if structure_label not in STRUCTURE_LIBRARY:
        raise ValueError(f"Unsupported structure_label: {structure_label}")

    if structure_label in {"graphene", "monolayer_hbn", "monolayer_mos2", "monolayer_cri3"}:
        if structure_label == "graphene":
            structure = build_graphene(hints)
        elif structure_label == "monolayer_hbn":
            structure = build_monolayer_hbn(hints)
        elif structure_label == "monolayer_mos2":
            structure = build_monolayer_mos2(hints)
        elif structure_label == "monolayer_cri3":
            structure = build_monolayer_cri3(hints)
        dimensionality = "2D"

    else:
        if not chemical_formula:
            raise ValueError("No chemical formula inferred for 3D prototype builder.")

        elements = list(Composition(chemical_formula).get_el_amt_dict().keys())

        if structure_label == "fcc":
            structure = build_fcc(chemical_formula, hints)
        elif structure_label == "bcc":
            structure = build_bcc(chemical_formula, hints)
        elif structure_label == "diamond":
            structure = build_diamond(chemical_formula, hints)
        elif structure_label == "hcp":
            structure = build_hcp(chemical_formula, hints)
        elif structure_label == "rocksalt":
            structure = build_rocksalt(elements, hints)
        elif structure_label == "zincblende":
            structure = build_zincblende(elements, hints)
        elif structure_label == "wurtzite":
            structure = build_wurtzite(elements, hints)
        elif structure_label == "perovskite":
            structure = build_perovskite(elements, hints)
        elif structure_label == "rutile":
            structure = build_rutile(elements, hints)
        elif structure_label == "anatase":
            structure = build_anatase(elements, hints)
        else:
            raise ValueError(f"No deterministic builder available for {structure_label}")

    vacuum_padding_z = 0.0
    if dimensionality == "2D":
        vacuum_padding_z = 18.0
        structure = add_vacuum_to_structure(structure, vacuum_padding_z)

    meta = {
        "structure_label": structure_label,
        "structure_family": structure_family,
        "space_group": STRUCTURE_LIBRARY[structure_label]["space_group"],
        "crystal_system": STRUCTURE_LIBRARY[structure_label]["crystal_system"],
        "dimensionality": dimensionality,
        "vacuum_padding_z": vacuum_padding_z,
    }
    return structure, meta


# ============================================================
# 8. Candidate / ambiguity output
# ============================================================

def make_candidate_output(parsed: Dict[str, Any], user_input: str) -> Dict[str, Any]:
    formula = parsed.get("chemical_formula")
    elements = parsed.get("elements", [])
    structure_label = parsed.get("structure_label", "unknown")
    structure_family = parsed.get("structure_family", "unknown")

    candidates = []
    if formula:
        candidates.append({
            "chemical_formula": formula,
            "structure_label_hint": structure_label,
            "structure_family_hint": structure_family,
            "reason": "Formula inferred, but no deterministic builder is available for this description."
        })
    elif elements:
        candidates.append({
            "elements": elements,
            "structure_label_hint": structure_label,
            "structure_family_hint": structure_family,
            "reason": "Elements identified, but composition/structure is underdetermined."
        })

    return {
        "status": "candidate_only",
        "user_description": user_input,
        "material_metadata": {
            "chemical_formula": formula,
            "crystal_system": "unknown",
            "space_group": "unknown",
            "structure_family": structure_family,
            "structure_label": structure_label,
            "construction_method": "candidate_identification_only",
            "notes": parsed.get("notes", ""),
        },
        "lattice_parameters": {},
        "atomic_positions": [],
        "dft_computation_hints": {
            "dimensionality": parsed.get("dimensionality", "unknown"),
            "vacuum_padding_z": 0.0,
            "is_spin_polarized": bool(parsed.get("is_spin_polarized", False)),
            "magnetic_state": parsed.get("magnetic_state", "unknown"),
            "exchange_correlation": "PBE",
            "include_soc": bool(parsed.get("include_soc", False)),
            "hubbard_u_values": {},
            "calculation_task": parsed.get("calculation_task", "default"),
        },
        "candidates": candidates,
        "validation": {
            "confidence": parsed.get("confidence", 0.0),
            "message": "Description was partially understood, but deterministic structure generation was not possible.",
        }
    }


def make_ambiguous_output(parsed: Dict[str, Any], user_input: str) -> Dict[str, Any]:
    return {
        "status": "ambiguous",
        "user_description": user_input,
        "material_metadata": {
            "chemical_formula": parsed.get("chemical_formula"),
            "crystal_system": "unknown",
            "space_group": "unknown",
            "structure_family": parsed.get("structure_family", "unknown"),
            "structure_label": parsed.get("structure_label", "unknown"),
            "construction_method": "none",
            "notes": parsed.get("notes", ""),
        },
        "lattice_parameters": {},
        "atomic_positions": [],
        "dft_computation_hints": {
            "dimensionality": parsed.get("dimensionality", "unknown"),
            "vacuum_padding_z": 0.0,
            "is_spin_polarized": bool(parsed.get("is_spin_polarized", False)),
            "magnetic_state": parsed.get("magnetic_state", "unknown"),
            "exchange_correlation": "PBE",
            "include_soc": bool(parsed.get("include_soc", False)),
            "hubbard_u_values": {},
            "calculation_task": parsed.get("calculation_task", "default"),
        },
        "validation": {
            "confidence": parsed.get("confidence", 0.0),
            "message": "Input description is too ambiguous for reliable structure generation.",
        }
    }


# ============================================================
# 9. Convert structure to final JSON
# ============================================================

def structure_to_output_json(
    structure: Structure,
    parsed: Dict[str, Any],
    meta: Dict[str, Any],
    user_input: str
) -> Dict[str, Any]:
    lattice = structure.lattice

    atomic_positions = []
    for site in structure.sites:
        atomic_positions.append({
            "element": site.species_string,
            "coordinate_type": "fractional",
            "position": [
                round(float(site.frac_coords[0]), 8),
                round(float(site.frac_coords[1]), 8),
                round(float(site.frac_coords[2]), 8),
            ]
        })

    return {
        "status": "fully_supported",
        "user_description": user_input,
        "material_metadata": {
            "chemical_formula": structure.composition.reduced_formula,
            "crystal_system": meta["crystal_system"],
            "space_group": meta["space_group"],
            "structure_family": meta["structure_family"],
            "structure_label": meta["structure_label"],
            "construction_method": "deterministic_builder",
            "notes": parsed.get("notes", ""),
        },
        "lattice_parameters": {
            "a": round(float(lattice.a), 8),
            "b": round(float(lattice.b), 8),
            "c": round(float(lattice.c), 8),
            "alpha": round(float(lattice.alpha), 8),
            "beta": round(float(lattice.beta), 8),
            "gamma": round(float(lattice.gamma), 8),
        },
        "atomic_positions": atomic_positions,
        "dft_computation_hints": {
            "dimensionality": meta["dimensionality"],
            "vacuum_padding_z": float(meta["vacuum_padding_z"]),
            "is_spin_polarized": bool(parsed.get("is_spin_polarized", False)),
            "magnetic_state": parsed.get("magnetic_state", "unknown"),
            "exchange_correlation": "PBE",
            "include_soc": bool(parsed.get("include_soc", False)),
            "hubbard_u_values": {},
            "calculation_task": parsed.get("calculation_task", "geometry_optimization"),
        },
        "validation": {
            "confidence": parsed.get("confidence", 0.0),
            "num_sites": len(structure.sites),
            "is_ordered": structure.is_ordered,
            "formula_from_structure": structure.composition.formula,
        }
    }


# ============================================================
# 10. Main pipeline
# ============================================================

def sanitize_filename(text: str, max_len: int = 48) -> str:
    text = text.strip().replace(" ", "_").replace("'", "")
    text = re.sub(r"[^A-Za-z0-9_\-]", "", text)
    return text[:max_len] or "material"


def run_structure(user_input: Optional[str] = None) -> Optional[str]:
    if not user_input:
        print("-" * 60)
        user_input = input("Please describe your material system:\n> ").strip()
        if not user_input:
            user_input = "wurtzite gallium nitride"
            print(f"Empty input detected, using default test case: '{user_input}'")

    print(f"\n🚀 Interpreting description: '{user_input}'")

    try:
        parsed = call_gemini_parser(user_input)
        print("✅ Gemini parsing succeeded.")
    except Exception as e:
        print(f"⚠️ Gemini parsing failed, using fallback parser. Reason: {e}")
        parsed = fallback_parse(user_input)

    parsed["structure_label"] = (parsed.get("structure_label") or "unknown").lower()
    parsed["structure_family"] = parsed.get("structure_family") or "unknown"
    parsed["dimensionality"] = parsed.get("dimensionality") or "unknown"
    parsed["magnetic_state"] = parsed.get("magnetic_state") or "unknown"

    if not parsed.get("chemical_formula"):
        parsed["chemical_formula"] = infer_formula(
            user_input,
            parsed.get("elements", []),
            parsed.get("structure_label", "unknown"),
        )

    structure_label = parsed.get("structure_label")
    chemical_formula = parsed.get("chemical_formula")

    try:
        if structure_label in STRUCTURE_LIBRARY and (
            chemical_formula is not None or structure_label in {"graphene", "monolayer_hbn", "monolayer_mos2", "monolayer_cri3"}
        ):
            structure, meta = build_supported_structure(parsed)
            output_data = structure_to_output_json(structure, parsed, meta, user_input)

        elif chemical_formula or parsed.get("elements"):
            output_data = make_candidate_output(parsed, user_input)

        else:
            output_data = make_ambiguous_output(parsed, user_input)

    except Exception as e:
        print(f"⚠️ Deterministic build failed: {e}")
        if chemical_formula or parsed.get("elements"):
            output_data = make_candidate_output(parsed, user_input)
            output_data["validation"]["builder_error"] = str(e)
        else:
            output_data = make_ambiguous_output(parsed, user_input)
            output_data["validation"]["builder_error"] = str(e)

    output_filename = f"{sanitize_filename(user_input)}.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Output saved as: {output_filename}")
    print(f"   Status: {output_data.get('status', 'unknown')}")
    print(f"   Formula: {output_data.get('material_metadata', {}).get('chemical_formula')}")

    if output_data.get("status") == "fully_supported":
        print(f"   Sites: {output_data.get('validation', {}).get('num_sites', 'N/A')}")
    else:
        print(f"   Message: {output_data.get('validation', {}).get('message', '')}")

    return output_filename


if __name__ == "__main__":
    run_structure()