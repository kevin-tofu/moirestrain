Tutorials
=========

The ``tutorials/`` directory contains small runnable workflows. They are meant
to be read and executed from the repository root.

.. code-block:: bash

   PYTHONPATH=src python tutorials/01_square_grid_analysis.py
   PYTHONPATH=src python tutorials/02_partial_grid_detection.py
   PYTHONPATH=src python tutorials/03_four_corner_rectification.py

Outputs are written under ``data/tutorials/``.

01. Square-Grid Analysis
------------------------

``tutorials/01_square_grid_analysis.py`` creates a front-facing square-marker
grid pair, analyzes x/y displacement and strain, saves arrays, and writes a
summary figure.

This is the smallest complete workflow for ``analyze_grid``.

02. Partial-Grid Detection
--------------------------

``tutorials/02_partial_grid_detection.py`` demonstrates the workflow where a
larger image contains only one grating patch. It detects the grating ROI,
analyzes the detected region, and compares measured strain against a synthetic
truth field.

The generated README figure comes from this workflow:

.. image:: _static/partial_grid_strain_measured_true.png
   :alt: Partial-grid ROI detection and measured/true strain comparison
   :width: 100%

03. Four-Corner Rectification
-----------------------------

``tutorials/03_four_corner_rectification.py`` creates a tilted planar grating
view, rectifies it from four image-space corner points, and then runs grid
analysis on the rectified pair. This is the recommended pattern when the camera
is tilted relative to the target surface and the four grating corners are
available.
