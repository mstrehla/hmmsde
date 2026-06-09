# HMMSDE
Implementation of the HMMSDE method from the paper "[Likelihood-Based Estimation of Multidimensional Langevin Models and Its Application to Biomolecular Dynamics](https://publications.imp.fu-berlin.de/39/1/HoSc06.pdf)" by Horenko & Schütte. 
This implementation uses a Langevin model with a Müller-Brown potential for the drift to generate samples from. 

Call example
-------

    pip install numpy scipy matplotlib
    python run_experiment.py --T-phys 2000 --K-prime 6 --seed 42
