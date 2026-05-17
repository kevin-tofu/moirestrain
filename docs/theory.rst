Theory notes
============

Sampling moire phase
--------------------

The sampling moire method generates multiple phase-shifted moire images from a
single grating image by thinning the image at different sampling offsets and
interpolating each sampled image back to the original grid.

For equally shifted images

.. math::

   I_k = a + b \cos\left(\phi + \frac{2\pi k}{N}\right),

the wrapped phase can be estimated from the first Fourier component:

.. math::

   \phi = \mathrm{atan2}\left(-\sum_k I_k \sin\frac{2\pi k}{N},
                              \sum_k I_k \cos\frac{2\pi k}{N}\right).

In ``moirestrain``, ``phase_shifted_stack`` generates the phase-shifted moire
image stack and ``wrapped_phase`` applies this phase-shifting formula. The
high-level entry point ``phase_shifted_sampling_moire`` runs the complete
single-direction workflow.

Square-marker grid targets
--------------------------

For a square-marker grid target, the captured image is treated as a grayscale
image after camera pixel averaging. ``separate_grid_components`` first applies
directional smoothing: smoothing along y isolates the x-periodic component,
and smoothing along x isolates the y-periodic component. ``analyze_grid`` then
applies the phase-shifted sampling moire workflow to each component and
computes the small-strain field.

Displacement from phase difference
----------------------------------

The displacement component aligned with the analyzed grating direction is
calculated from the phase difference between deformed and reference images:

.. math::

   u = \frac{\Delta\phi}{2\pi} p.

Here ``p`` is the grating pitch in the displacement unit of interest. If the
pitch is expressed in pixels, the displacement is returned in pixels.

Strain field direction
----------------------

Small strain is obtained from spatial gradients of displacement:

.. math::

   \varepsilon_{xx} = \frac{\partial u}{\partial x}, \quad
   \varepsilon_{yy} = \frac{\partial v}{\partial y}, \quad
   \gamma_{xy} = \frac{\partial u}{\partial y} + \frac{\partial v}{\partial x}.

``moirestrain.strain_field`` provides these small-strain components. For noisy
experimental data, use the ``smooth_window`` option because differentiating a
noisy displacement field directly amplifies high-frequency noise.

Perspective rectification
-------------------------

For a planar specimen surface, camera tilt can be modeled by a homography:

.. math::

   \mathbf{x}' \sim \mathbf{H}\mathbf{x}.

``moirestrain.rectify_image`` estimates this transform from four image corner
points and resamples the grating ROI into a fronto-parallel image. Phase,
displacement, and strain analysis should then be performed in the rectified
coordinate system. If physical corner coordinates are available,
``PerspectiveCalibration.pixel_spacing`` can be passed to ``strain_field`` as
``spacing=(dy, dx)``.
