pip install -e .
python3 alabEBM/tests/test.py

rm -rf alabEBM/tests/soft_kmeans alabEBM/tests/hard_kmeans alabEBM/tests/conjugate_priors
# Move directories from current directory instead of root
[ -d soft_kmeans ] && mv soft_kmeans alabEBM/tests/
[ -d hard_kmeans ] && mv hard_kmeans alabEBM/tests/
[ -d conjugate_priors ] && mv conjugate_priors alabEBM/tests/