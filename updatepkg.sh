# Clean up old build artifacts
rm -rf dist/
rm -rf build/
rm -rf alabebm.egg-info/

# Build the package using the build module
python -m pip install --upgrade build
python -m build

# Upload using twine
python -m pip install --upgrade twine
python -m twine upload dist/*
