# 2DSFS-scan
Python scripts for inference of genomic divergence between populations using the joint site frequency spectrum (SFS) and composite likelihoods. Our approach aims to detect aberrant 2D SFS relative to the background genome. The pipeline uses VCF and population map files as input to extract SNP data, computes the 1D and 2D SFS, and calculate likelihood statistics in genomic windows of a fixed size.

## Scripts
- 'scripts/src/twoDSFS_class.py' has the functions used for inference using empirical data.
- 'scripts/sims_scan.py' has the functions used for inference using simulated data.

adding a small update LHU
