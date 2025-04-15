pip install -e .
# rm -rf mle em hard_kmeans conjugate_priors kde
# rm -rf pysaebm/test/mle pysaebm/test/em pysaebm/test/hard_kmeans pysaebm/test/conjugate_priors pysaebm/test/kde
# python3 pysaebm/test/test.py
# # Move directories from current directory instead of root
# [ -d mle ] && mv mle pysaebm/test/
# [ -d hard_kmeans ] && mv hard_kmeans pysaebm/test/
# [ -d conjugate_priors ] && mv conjugate_priors pysaebm/test/
# [ -d em ] && mv em pysaebm/test/
# [ -d kde ] && mv kde pysaebm/test/

rm -rf algo_results pysaebm/test/algo_results
python3 pysaebm/test/test.py
[ -d algo_results ] && mv algo_results pysaebm/test/