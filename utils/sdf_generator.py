from openbabel import pybel
import os
from pathlib import Path

dir=(r'./xyz')
files = list(os.listdir(dir))
# for i in files: print(i)
# for i in files: 
#     print(Path(i).stem)
# mol1 = next(pybel.readfile("xyz", "./xyz/0.xyz"))
mols = [next(pybel.readfile("xyz", f"./xyz/{i}")) for i in files]

# print(mols)

# # mol1.write("sdf", "./sdf/0.sdf")

try:
    for file, mol in zip(files, mols):
        mol.write("sdf", f"./sdf_2/{Path(file).stem}.sdf")
        print(f"Converted {file}.xyz to {Path(file).stem}.sdf") 
except Exception as e:
    print(f"Error: {e} for {i}.xyz")    


