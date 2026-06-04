#!/bin/bash
echo "Activating DLENV-1 Environment"
source activate dlenv-1
python --version
echo "Launching Alpha script"
python Perf_compa_2D_alpha.py
echo " Alpha script executed successfully"
reset
echo "$CONDA_DEFAULT_ENV"
python --version
echo "Launching Beta script"
python Perf_compa_2D_beta.py
echo " Beta script executed successfully"
reset
echo "$CONDA_DEFAULT_ENV"
python --version
echo "Launching Gamma script"
python Perf_compa_2D_gamma.py
echo " Gamma script executed successfully"
reset
echo "$CONDA_DEFAULT_ENV"
python --version
echo "Launching Delta script"
python Perf_compa_2D_delta.py
echo " Delta script executed successfully"
reset
echo "$CONDA_DEFAULT_ENV"
python --version
echo "Launching Epsilon script"
python Perf_compa_2D_epsilon.py
echo " Epsilon script executed successfully"
reset
echo "$CONDA_DEFAULT_ENV"
python --version
echo "Launching Zeta script"
python Perf_compa_2D_zeta.py
echo " Zeta script executed successfully"
reset
echo "$CONDA_DEFAULT_ENV"
python --version
echo "Launching Eta script"
python Perf_compa_2D_eta.py
echo " Eta script executed successfully"
