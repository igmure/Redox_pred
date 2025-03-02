import os
from typing import Dict, Tuple, Optional
import pandas as pd
from utils.DatasetToCSVBase import DatasetToCSVBaseClass
from pathlib import Path


class DatasetToCSVROP313Class_sdf(DatasetToCSVBaseClass):
    """
    Implementation of the DatasetToCSVBase class that parses XYZ and related files for the ROP313 dataset,
    extracts relevant data (dG_red, solvent type, charge, unpaired electron count), 
    and processes it into a structured format for CSV export.
    """

    def parse_floatvalue_file(self, file_path: str) -> Optional[float]:
        """
        Parses the float-value file.

        Args:
        - file_path: Path to the file with only float value.

        Returns:
        - Float value if found, None otherwise.
        """
        try:
            with open(file_path, "r") as file:
                value = float(file.read().strip())
            return value
        except Exception as e:
            print(f"Error parsing file {file_path}: {e}")
        return None

    def parse_strvalue_file(self, file_path: str) -> Optional[str]:
        """
        Parses the string-value file.

        Args:
        - file_path: Path to the file with only string value.

        Returns:
        - String value if found, None otherwise.
        """
        try:
            with open(file_path, "r") as file:
                value = file.read().strip()
            return value
        except Exception as e:
            print(f"Error parsing .solv file {file_path}: {e}")
        return None

    def parse_intvalue_file(self, file_path: str) -> Optional[int]:
        """
        Parses the integer-value file.

        Args:
        - file_path: Path to the file with only int value.

        Returns:
        - Integer value if found, None otherwise.
        """
        try:
            with open(file_path, "r") as file:
                value = int(file.read().strip())
            return value
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
        return None

    def parse_xyz_file(self, file_path: str, charge: int) -> Tuple[Optional[int], Optional[str]]:
        """
        Parses the XYZ file for geometry data.

        Args:
        - file_path: Path to the XYZ file.

        Returns:
        - A tuple containing:
            - num_atoms: The number of atoms in the structure (or None if invalid).
            - smiles: The SMILES string (or None if conversion fails).
        """
        try:
            with open(file_path, "r") as file:
                xyz_content = file.readlines()

            num_atoms = int(xyz_content[0].strip())
            smiles = self.xyz_to_smiles(file_path, charge)

            return num_atoms, smiles
        except (ValueError, IndexError) as e:
            print(f"Error parsing XYZ file {file_path}: {e}")
        return None, None

    def extract_metadata(self, folder_name: str) -> Tuple[str, str, int, int, int, int]:
        """
        Extracts metadata (system number, and solvent type) from the folder structure.

        Args:
        - folder_name: The folder name.

        Returns:
        - A tuple containing:
            - system_number: The system number within the family.
            - solvent_type: The solvent type (from .solv file).
            - charge_gn: charge of ground-state mol (from .CHRG1 file)
            - charge_rd: charge of reduced mol (from .CHRG2 file)
            - uhf_gn: unpaired electrons in ground-state (from .UHF1 file)
            - uhf_rd: unpaired electrons in reduced state (from .UHF2 file)
        """
        system_number = folder_name.split('/')[-1].strip()

        solv_file_path = os.path.join(folder_name, ".solv")
        ref_file_path = os.path.join(folder_name, ".ref")
        chrg_file_path_1 = os.path.join(folder_name, ".CHRG1")
        chrg_file_path_2 = os.path.join(folder_name, ".CHRG2")
        uhf_file_path_1 = os.path.join(folder_name, ".UHF1")
        uhf_file_path_2 = os.path.join(folder_name, ".UHF2")

        solvent_type = self.parse_strvalue_file(solv_file_path)
        dG_red = self.parse_floatvalue_file(ref_file_path)
        charge_gn = self.parse_intvalue_file(chrg_file_path_1)
        charge_rd = self.parse_intvalue_file(chrg_file_path_2)
        uhf_gn = self.parse_intvalue_file(uhf_file_path_1)
        uhf_rd = self.parse_intvalue_file(uhf_file_path_2)

        return system_number, solvent_type, dG_red, charge_gn, charge_rd, uhf_gn, uhf_rd

    def process_files(self, folder_path: str) -> pd.DataFrame:
        """
        Processes all folder data and returns a DataFrame with data for each system.

        Args:
        - folder_path: Path to the folder containing the XYZ and related files.

        Returns:
        - A pandas DataFrame containing data for each system with columns:
          - dG_red, solvent_type, charge_gn, charge_rd, uhf_gn, uhf_rd, SMILES, etc.
        """
        data = {}

        # Iterate through all subfolders in the main folder
        if os.path.isdir(folder_path):
            
            sdfs=list(os.listdir(folder_path))
            for file in sdfs:
                full_path=folder_path+'/'+file
                # Parse XYZ geometries
                sdf = self.sdf_to_mol(full_path)
                system_number = file[:-4]

            # Store data for each system
                data[system_number] = {
                    "system_number": Path(file).stem,
                    "sdf": sdf,
                }

        # Convert the dictionary to a DataFrame
        rows = []
        for system_number, values in data.items():
            row = {
                "system_number": system_number,
                "sdf":values["sdf"],
            }
            rows.append(row)

        return pd.DataFrame(rows)
