import json
import os

import structure
import generate


def main():
    print("=" * 60)
    print("      Welcome to the Unified LLM-DFT Framework")
    print("=" * 60)
    print("\nThis tool converts natural-language descriptions into")
    print("DFT-ready structured data and input files.\n")

    # Step 1: Get user input
    user_input = input("Please describe your material system\n").strip()

    if not user_input:
        user_input = "wurtzite gallium nitride"
        print(f"\nEmpty input detected, using default test case: '{user_input}'")

    # Step 2: Run structure generation
    generated_json = structure.run_structure(user_input)

    if not generated_json:
        print("\n❌ Pipeline aborted because structure generation failed.")
        return

    if not os.path.exists(generated_json):
        print(f"\n❌ Pipeline aborted because JSON file was not found: {generated_json}")
        return

    # Step 3: Inspect structure status before calling generate.py
    try:
        with open(generated_json, "r", encoding="utf-8") as f:
            material_data = json.load(f)
    except Exception as e:
        print(f"\n❌ Pipeline aborted because the generated JSON could not be read: {e}")
        return

    status = material_data.get("status", "unknown")

    if status != "fully_supported":
        print(f"\n⚠️ Structure stage completed, but status is '{status}'.")
        print("DFT input generation will be skipped.")
        print(material_data.get("validation", {}).get("message", "No detailed structure available."))
        return

    # Step 4: Run input-file generation
    print("\nProceeding to DFT input generation...")
    generate.run_generate(json_filename=generated_json)


if __name__ == "__main__":
    main()
