pip install -e .
# rm -rf mle em hard_kmeans conjugate_priors kde
# rm -rf sa_ebm/test/mle sa_ebm/test/em sa_ebm/test/hard_kmeans sa_ebm/test/conjugate_priors sa_ebm/test/kde
# python3 sa_ebm/test/test.py
# # Move directories from current directory instead of root
# [ -d mle ] && mv mle sa_ebm/test/
# [ -d hard_kmeans ] && mv hard_kmeans sa_ebm/test/
# [ -d conjugate_priors ] && mv conjugate_priors sa_ebm/test/
# [ -d em ] && mv em sa_ebm/test/
# [ -d kde ] && mv kde sa_ebm/test/

rm -rf algo_results sa_ebm/test/algo_results
python3 sa_ebm/test/test.py
[ -d algo_results ] && mv algo_results sa_ebm/test/