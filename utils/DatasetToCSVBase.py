from abc import ABC, abstractmethod
from typing import Tuple, Optional
from rdkit import Chem
import pandas as pd


class DatasetToCSVBaseClass(ABC):
    """
    Abstract base class for handling XYZ files and processing them into CSV-compatible data.
    """

    def xyz_to_smiles(self, file_path: str) -> Optional[str]:
        """
        Converts the content of an XYZ file to SMILES format.

        Args:
        - file_path: Path to the XYZ file.

        Returns:
        - SMILES string if conversion is successful, None otherwise.
        """
        try:
            with open(file_path, "r") as file:
                xyz_content = file.read()

            mol = Chem.MolFromXYZBlock(xyz_content)
            if mol:
                return Chem.MolToSmiles(mol)
        except Exception as e:
            print(f"Error converting XYZ to SMILES for file {file_path}: {e}")
        return None

    @abstractmethod
    def parse_xyz_file(
        self, file_path: str
    ) -> Tuple[Optional[int], Optional[float], Optional[float], Optional[str]]:
        """
        Parses the XYZ file to extract the number of atoms, SCF energy (E), Gibbs free energy (G), and the SMILES string.

        Args:
        - file_path: Path to the XYZ file.

        Returns:
        - A tuple containing:
            - num_atoms: The number of atoms in the structure (or None if invalid).
            - e_value: The SCF energy (E) value (or None if not found).
            - g_value: The Gibbs free energy (G) value (or None if not found).
            - smiles: The SMILES string (or None if conversion fails).
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def process_files(self, folder_path: str) -> pd.DataFrame:
        """
        Processes all XYZ files in the folder and returns a DataFrame with data for each family-system pair.

        Args:
        - folder_path: Path to the folder containing the XYZ files.

        Returns:
        - A pandas DataFrame containing data for each family-system pair, with separate columns for
          SCF energy (E) and Gibbs free energy (G) for each structure type (gn, ox, rd).
        """
        pass
