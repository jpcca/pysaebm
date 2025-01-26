rm -rf dist/
rm -rf build/
rm -rf alaEBM.egg-info/
python setup.py sdist bdist_wheel
twine upload dist/*
