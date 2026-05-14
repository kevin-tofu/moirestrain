Project scope
=============

``moirestrain`` is intended for optical metrology workflows that use a periodic
grating pattern on a specimen or target surface.

Primary goals
-------------

* Estimate phase fields from grating images using the sampling moire method.
* Calculate displacement fields from phase differences.
* Provide strain-field utilities for small deformation measurement.
* Rectify planar grating regions before phase analysis when the camera is
  tilted relative to the specimen surface.
* Keep the numerical core NumPy-first, while supporting practical CLI image I/O
  and PNG reporting.
* Expose an API that works naturally in scripts, notebooks, and automated
  experimental pipelines.

Relationship to other tools
---------------------------

``moirestrain`` is not a Digital Image Correlation package. DIC packages estimate
displacement by correlating texture or speckle patterns. ``moirestrain`` uses
periodic grating phase information.

It is also not a moire lattice simulation package. Packages such as
``moirepy`` target atomistic moire lattice modeling. ``moirestrain`` targets
image-based displacement and strain measurement.

Phase unwrapping packages can still be useful companions. The default
implementation uses NumPy's unwrap functionality, while future versions may
offer adapters for more robust 2D unwrapping algorithms.
