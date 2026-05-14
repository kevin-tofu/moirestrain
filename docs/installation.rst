Installation
============

Development install
-------------------

Clone the repository and install dependencies with Poetry:

.. code-block:: bash

   poetry install

Run tests:

.. code-block:: bash

   poetry run pytest

Build the source distribution and wheel:

.. code-block:: bash

   poetry build

Dependencies
------------

Runtime dependencies are intentionally small:

* NumPy

Development dependencies include:

* pytest
* pytest-cov
* scikit-image
* Sphinx and documentation extensions

``scikit-image`` is used for integration tests and examples that need image
warping or noise generation. It is not required by the core runtime API.
