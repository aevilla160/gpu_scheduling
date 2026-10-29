#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mkdir -p data/raw
curl --fail --location --proto '=https' https://snap.stanford.edu/data/ca-GrQc.txt.gz -o data/raw/ca-GrQc.txt.gz
curl --fail --location --proto '=https' https://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/bcsstruc2/bcsstk17.mtx.gz -o data/raw/bcsstk17.mtx.gz
python3 prepare_data.py
