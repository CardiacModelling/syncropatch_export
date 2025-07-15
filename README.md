[![Unit tests](https://github.com/CardiacModelling/syncropatch_export/actions/workflows/pytest.yml/badge.svg)](https://github.com//CardiacModelling/syncropatch_export/actions/workflows/pytest.yml)
[![codecov](https://codecov.io/gh/CardiacModelling/syncropatch_export/graph/badge.svg?token=HOL0FrpGqs)](https://codecov.io/gh/CardiacModelling/syncropatch_export)

This repository contains a python package and scripts for processing data outputted from Nanion SynroPatch 384.

With this package you can export each sweep of each protocol for each well as individual files (.csv). 
Meta-data describing the protocol, and variables such as membrance capacitance (Cm), Rseries and Rseal can be exported.

This package is tested on Ubuntu with Python 3.8, 3.9, 3.10, 3.11, 3.12, and 3.13 and is distributed under a [BSD 3-Clause License](./LICENSE).

## Getting Started

First clone this repository

```sh
git clone git@github.com:CardiacModelling/syncropatch_export 
cd syncropatch_export
```

Create and activate a virtual environment.

```sh
python3 -m venv .venv && source .venv/bin/activate
```

Then install the package with `pip`.

```
python3 -m pip install --upgrade pip && python3 -m pip install -e .'[test]'
```

To run the tests you must first download some test data.
Test data is available at [cardiac.nottingham.ac.uk/syncropatch\_export](https://cardiac.nottingham.ac.uk/syncropatch_export)

```
wget https://cardiac.nottingham.ac.uk/syncropatch_export/test_data.tar.xz -P tests/
tar xvf tests/test_data.tar.xz -C tests/
rm tests/test_data.tar.xz
```

Then you can run the tests.
```
python3 -m unittest
```

## Usage example

...TODO


## Development

Commits should be merged in via pull requests.

Tests are written using the standard [unittest](https://docs.python.org/3.13/library/unittest.html) framework.

Online testing, style-checking, and coverage testing is set up using GitHub actions.
Coverage testing is handled via [Codecov](https://about.codecov.io/).

Documentation is implemented using [Sphinx](https://www.sphinx-doc.org/).
To compile locally, first install the required dependencies
```
pip install -e .'[docs]'
```
and then use Make
```
cd docs
make html
```


