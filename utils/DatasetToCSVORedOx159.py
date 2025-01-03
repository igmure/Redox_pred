import os
from typing import Dict, Tuple, Optional
import pandas as pd
from utils.DatasetToCSVBase import DatasetToCSVBaseClass

class DatasetToCSVORO159Class(DatasetToCSVBaseClass):
    """
    Implementation of the DatasetToCSVBase class that parses XYZ files for ORedOX159 dataset,
    extracts relevant data, and processes it into a structured format for CSV export.
    """

    def parse_xyz_file(
        self, file_path: str, charge: int
    ) -> Tuple[Optional[int], Optional[float], Optional[float], Optional[str]]:
        """
        Parses the XYZ file to extract the number of atoms, SCF energy (E), Gibbs free energy (G), and the SMILES string.

        Args:
        - file_path: Path to the XYZ file.
        - charge: Charge of the system.

        Returns:
        - A tuple containing:
            - num_atoms: The number of atoms in the structure (or None if invalid).
            - e_value: The SCF energy (E) value (or None if not found).
            - g_value: The Gibbs free energy (G) value (or None if not found).
            - smiles: The SMILES string (or None if conversion fails).
        """
        try:
            with open(file_path, "r") as file:
                xyz_content = file.readlines()

            # Extract number of atoms
            num_atoms = int(xyz_content[0].strip())

            # Extract SCF energy (E) and Gibbs free energy (G)
            second_line = xyz_content[1].strip()
            e_value = (
                float(second_line.split("E:")[1].split()[0])
                if "E:" in second_line
                else None
            )
            g_value = (
                float(second_line.split("G:")[1].split()[0])
                if "G:" in second_line
                else None
            )

            # Call xyz_to_smiles function to get the SMILES string
            smiles = self.xyz_to_smiles(file_path, charge=charge)

            return num_atoms, e_value, g_value, smiles

        except (ValueError, IndexError) as e:
            print(f"Error parsing XYZ file {file_path}: {e}")
            return None, None, None, None

    def extract_metadata(self, filename: str) -> Tuple[str, str, int, str]:
        """
        Extracts metadata (family, system number, charge, and structure type) from the filename.

        Args:
        - filename: The XYZ file name.

        Returns:
        - A tuple containing:
            - family: The family name extracted from the filename.
            - system: The system number within the family.
            - charge: The charge of the system.
            - structure_type: The type of the structure (gn, ox, rd).
        """
        parts = filename.replace(".xyz", "").split("_")
        family, system_number = parts[0].split("-")

        charge_mapping = {"0n": 0, "1a": -1, "2a": -2, "1c": 1, "2c": 2}

        charge = charge_mapping.get(parts[1], 0)
        structure_type = parts[2]

        return family, system_number, charge, structure_type

    def process_files(self, folder_path: str) -> pd.DataFrame:
        """
        Processes all XYZ files in the folder and returns a DataFrame with data for each family-system pair.

        Args:
        - folder_path: Path to the folder containing the XYZ files.

        Returns:
        - A pandas DataFrame containing data for each family-system pair, with separate columns for
          SCF energy (E) and Gibbs free energy (G) for each structure type (gn, ox, rd).
        """
        data: Dict[Tuple[str, str], Dict[int, Dict[str, Optional[float]]]] = {}

        G_ELECTRON = 0.864784  # Free Gibbs Energy for an electron [kcal/mol]

        # Iterate through all XYZ files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".xyz"):
                file_path = os.path.join(folder_path, filename)
                # Extract metadata from the filename
                (
                    family,
                    nFamily,
                    charge,
                    structure_type,
                ) = self.extract_metadata(filename)

                # Parse file data
                num_atoms, e_value, g_value, smiles = self.parse_xyz_file(
                    file_path, charge
                )


                # Initialize the data dictionary if the family-system pair doesn't exist
                if (family, nFamily) not in data:
                    data[(family, nFamily)] = {
                        "NumAtoms": num_atoms,
                        "SMILES": smiles,
                        "charge_gn": None,
                        "E_gn": None,
                        "G_gn": None,
                        "E_ox": None,
                        "G_ox": None,
                        "E_rd": None,
                        "G_rd": None,
                    }

                # Update data based on the structure type
                if structure_type == "gn":
                    data[(family, nFamily)]["charge_gn"] = charge

                e_key = f"E_{structure_type}"  # This will be gn, ox, or rd
                g_key = f"G_{structure_type}"
                data[(family, nFamily)][e_key] = e_value
                data[(family, nFamily)][g_key] = g_value

        # Convert the dictionary to a DataFrame
        rows = []
        for (family, nFamily), values in data.items():
            G_gn = values["G_gn"]
            G_ox = values["G_ox"]
            G_rd = values["G_rd"]

            # Calculate dG_red and dG_ox
            dG_red = None
            dG_ox = None

            if G_gn is not None and G_rd is not None:
                dG_red = G_rd - G_gn - G_ELECTRON

            if G_gn is not None and G_ox is not None:
                dG_ox = G_ox + G_ELECTRON - G_gn

            row = {
                "Family": family,
                "System": nFamily,
                "NumAtoms": values["NumAtoms"],
                "SMILES": values["SMILES"],
                "charge_gn": values["charge_gn"],
                "E_gn": values["E_gn"],
                "G_gn": values["G_gn"],
                "E_ox": values["E_ox"],
                "G_ox": values["G_ox"],
                "E_rd": values["E_rd"],
                "G_rd": values["G_rd"],
                "dG_red": dG_red,
                "dG_ox": dG_ox,
            }
            rows.append(row)

        return pd.DataFrame(rows)