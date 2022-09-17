#! /bin/bash
pip install -r requirements.txt
pip install -e fastar/baselines
pip install -e fastar/gym-midline
# pysmt-install --z3 --confirm-agreement      # For MACE
