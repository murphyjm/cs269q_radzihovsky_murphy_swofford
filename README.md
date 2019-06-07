# A QAOA Solution to the Traveling Salesman Problem using *pyQuil*
## CS 269Q Final Project
###  Matt Radzihovsky, Joey Murphy, Mason Swofford
Final Project Repository for CS 269Q, Spring 2019

QAOA TSP solution using mixers is implemented in ``tsp_qaoa_updated.py``, based on work presented in Hadfield *et. al* 2017 (paper included above).

QAOA constraint hamiltonian TSP solution is implemented in ``quantum.py``.

Classical solution implemented in ``classical.py``. See instructions below for installation.

# Installation instructions
After cloning this repository, users who already have a working installation of Anaconda can follow these steps to run the code:
1. Create a new conda environment
``conda create -n my_env``
2. Activate the environment
``source activate my_env``
3. Install pip
``conda install pip``
4. Install dependencies listed in the requirements.txt file
``pip install —user —requirement requirements.txt``
5. In another terminal window, activate the same virtual environment
``source activate my_env``
6. Initiate a QVM connection
``qvm -S``
6. You’re now ready to run the code in the original window. See files for command line argument documentation.
``python quantum.py 3``
``python tsp_qaoa_updated.py 3 0.75 1``


