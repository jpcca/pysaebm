pip install -e .
python3 alabebm/test/test.py

rm -rf alabebm/test/soft_kmeans alabebm/test/hard_kmeans alabebm/test/conjugate_priors
# Move directories from current directory instead of root
[ -d soft_kmeans ] && mv soft_kmeans alabebm/test/
[ -d hard_kmeans ] && mv hard_kmeans alabebm/test/
[ -d conjugate_priors ] && mv conjugate_priors alabebm/test/