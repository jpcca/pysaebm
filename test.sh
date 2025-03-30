pip install -e .
python3 alabebm/test/test.py

rm -rf alabebm/test/mle alabebm/test/em alabebm/test/hard_kmeans alabebm/test/conjugate_priors
# Move directories from current directory instead of root
[ -d mle ] && mv mle alabebm/test/
[ -d hard_kmeans ] && mv hard_kmeans alabebm/test/
[ -d conjugate_priors ] && mv conjugate_priors alabebm/test/
[ -d em ] && mv em alabebm/test/