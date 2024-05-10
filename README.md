# Mountaineer
Algorithm for posterior emulation using walkers exploring the posterior landscape

## Requirements
* Python3.8 or higher with NumPy, SciPy, Matplotlib and optionally Scikit-Learn (for GPR applications below) and Jupyter (for example usage).
* MLFundas: Basic building blocks of machine learning (ML) techniques [stored here](https://github.com/a-paranjape/mlfundas).
* Parameter inference framework [Cobaya](https://cobaya.readthedocs.io/en/) and analysis tool [GetDist](https://getdist.readthedocs.io/en/) (shipped with Cobaya).
* PICASA: Sampling and GPR training algorithm [stored here](https://bitbucket.org/aparanjape/picasa/). (We will only use GPR training for some examples.)

## Installation
* Clone into `mlfundas` [here](https://github.com/a-paranjape/mlfundas).
* Clone into `picasa` [here](https://bitbucket.org/aparanjape/picasa/).
* Edit `paths.py`, replacing `ML_Path` with the local path to `mlfundas/code/' and `Picasa_Path` with the local path to `picasa/code/`.
* Install Cobaya by following the instructions [here](https://cobaya.readthedocs.io/en/latest/installation.html).

## Contact
Aseem Paranjape: aseem_at_iucaa_dot_in
