pip install -e .
rm -rf mle em hard_kmeans conjugate_priors kde
rm -rf alabebm/test/mle alabebm/test/em alabebm/test/hard_kmeans alabebm/test/conjugate_priors alabebm/test/kde
python3 alabebm/test/test.py
# Move directories from current directory instead of root
[ -d mle ] && mv mle alabebm/test/
[ -d hard_kmeans ] && mv hard_kmeans alabebm/test/
[ -d conjugate_priors ] && mv conjugate_priors alabebm/test/
[ -d em ] && mv em alabebm/test/
[ -d kde ] && mv kde alabebm/test/