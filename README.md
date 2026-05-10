This is the GitHub repository for Josemaria Loza's Capstone Project in Mathematics at NYU Abu Dhabi, supervised by Sofiane Boaurroudj. 

The repository contains several folders, each with different output files or code scripts used during the project. The **main files** relevant to the project
correspond to **lss_{1,2,3,4}_op.py** for the verification of Burde's LSAs. The folder verification_burde also contains the Excel files for the explicit 
representations of the LSAs, as well as the associator computations for sl_2(k). 

For stages 1 and 2 of the superization, the entire code can be found in the **superization.py** file. To verify the results of this code, we ran a second code
using SageMath with Python. The code for this verification can be found in the **groebner_sage_osp12.ipynb** file.

For stage 2, in particular, the Nullstellensatz certificates for each polynomial system can be found in the **nullstellensatz_certs** file. 

The output_files folder is not relevant for code verification. 

All polynomials for each structure can be verified in the polys_files_sage folder. 

If you want to verify the computations, please make sure you have the SymPy library installed, as well as any other packages you might not have. 

To run the SageMath code, make sure to use an adequate environment/shell with Sage installed. 
