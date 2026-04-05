# %%


# %%
import numpy as np
from pyFAI.ext import splitBBox

# ---------------------------------------------------------------------------
# Module-level LUT cache for FiberIntegrator objects.
# Keys are the id() of the AzimuthalIntegrator/Transform object.
# Values are the promoted FiberIntegrator instances.
# The LUT inside each FiberIntegrator is built on the first integrate2d call
# and then reused automatically by pyFAI on all subsequent calls, giving
# ~9x speedup from the second call onward (~2s -> ~0.2s per image).
# ---------------------------------------------------------------------------
_fi_cache = {}


def _get_fiber_integrator(ai, incident_angle=0.0, tilt_angle=0.0, sample_orientation=4):
    """
    Return a FiberIntegrator for the given ai object, creating and caching
    it on first access. Subsequent calls with the same ai object return the
    cached instance instantly without any recomputation.

    Parameters:
    -----------
    :param ai: pyGIX Transform or pyFAI AzimuthalIntegrator
    :param incident_angle: incident angle in radians
    :param tilt_angle: tilt angle in radians
    :param sample_orientation: pyGIX sample orientation integer (1-4)
    """
    cache_key = id(ai)
    if cache_key not in _fi_cache:
        try:
            fi = ai.promote(type_="pyFAI.integrator.fiber.FiberIntegrator")
        except AttributeError:
            # Fallback for integrator objects that do not support promote()
            from pyFAI.integrator.fiber import FiberIntegrator
            fi = FiberIntegrator(dist=ai.dist,
                                 poni1=ai.poni1,
                                 poni2=ai.poni2,
                                 rot1=ai.rot1,
                                 rot2=ai.rot2,
                                 rot3=ai.rot3,
                                 wavelength=ai.wavelength,
                                 detector=ai.detector)
        _fi_cache[cache_key] = fi
    return _fi_cache[cache_key]


def remesh_gi(data, ai, npt=None, q_h_range=None, q_v_range=None, method='splitbbox', mask=None,
              incident_angle=0.0, tilt_angle=0.0, sample_orientation=4):
    """
    Redraw the Grazing-Incidence image in (qp, qz) coordinates using pyGIX.

    Uses pyFAI's FiberIntegrator with persistent LUT caching instead of
    calling ai.transform_reciprocal() directly. The LUT is built once on the
    first call per integrator object (~2s) and reused on all subsequent calls
    (~0.2s), giving ~9x speedup across a batch of images with fixed geometry.

    Parameters:
    -----------
    :param data: 2D image in pixel
    :type data: numpy 2D array of float
    :param ai: pyGIX transform operator
    :type ai: pyGIXTransform operator
    :param npt: number of points for the binning, as (npt_ip, npt_oop) tuple
    :type npt: tuple of int, optional
    :param q_h_range: Starting and ending point for the q_horizontal range
    :type q_h_range: Tuple(float, float), optional
    :param q_v_range: Starting and ending point for the q_vertical range
    :type q_v_range: Tuple(float, float), optional
    :param method: kept for API compatibility, FiberIntegrator uses 'splitpix' internally
    :type method: string
    :param mask: Mask of the 2D raw image
    :type mask: numpy 2D array of boolean
    :param incident_angle: grazing incident angle in radians
    :type incident_angle: float
    :param tilt_angle: tilt angle in radians
    :type tilt_angle: float
    :param sample_orientation: pyGIX sample orientation integer (1-4)
    :type sample_orientation: int
    """
    fi = _get_fiber_integrator(ai, incident_angle=incident_angle,
                               tilt_angle=tilt_angle,
                               sample_orientation=sample_orientation)

    if npt is None:
        npt_ip  = data.shape[1]
        npt_oop = data.shape[0]
    else:
        npt_ip, npt_oop = int(npt[0]), int(npt[1])

    result = fi.integrate2d_grazing_incidence(
        data=data.astype(float),
        npt_ip=npt_ip,
        npt_oop=npt_oop,
        sample_orientation=sample_orientation,
        incident_angle=incident_angle,
        tilt_angle=tilt_angle,
        unit_ip="qip_A^-1",
        unit_oop="qoop_A^-1",
        ip_range=q_h_range,
        op_range=q_v_range,
        method='splitpix',
        mask=mask
    )

    intensity, q_ip, q_oop = result
    return intensity, q_ip, q_oop


def remesh_transmission(image, ai, bins=None, q_h_range=None, q_v_range=None, mask=None):
    """
    Redraw the Transmission image in (qp, qz) coordinates using pyFAI splitBBox.histoBBox2d method

    Parameters:
    -----------
    :param image: 2D raw Detector image in pixel
    :type image: ndarray
    :param ai: PyFAI AzimuthalIntegrator
    :type ai: PyFAI AzimuthalIntegrator
    :param bins: number of point for the binning
    :type bins: int
    :param q_h_range: Starting and ending point for the q_horizontal range
    :type q_h_range: Tuple(float, float), optional
    :param q_v_range: Starting and ending point for the q_vertical range
    :type q_v_range: Tuple(float, float), optional
    :param mask: Mask of the 2D raw image
    :type mask: numpy 2D array of boolean
    """

    assert image.shape == ai.detector.shape
    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])
    px_x, px_y = np.meshgrid(x, y)
    r_z, r_y, r_x = ai.calc_pos_zyx(d1=px_y, d2=px_x)

    alphas = alpha(r_x, r_y, r_z)
    phis = phi(r_x, r_y, r_z)

    q_x, q_y, q_z = q_from_angles(phis, alphas, ai.wavelength) * 1e-10
    q_v = q_y
    q_h = q_x

    resc_q = False
    if -q_v.min() > np.pi:
        resc_q = True
        q_v *= 0.1
        q_h *= 0.1

    if bins is None: bins = tuple(reversed(image.shape))
    if q_h_range is None:
        q_h_range = (q_h.min(), q_h.max())
    if q_v_range is None:
        q_v_range = (q_v.min(), q_v.max())

    I, q_y, q_z, _, _ = splitBBox.histoBBox2d(weights=image,
                                              pos0=q_h,
                                              delta_pos0=np.ones_like(image) * (q_h_range[1] - q_h_range[0]) / bins[0],
                                              pos1=q_v,
                                              delta_pos1=np.ones_like(image) * (q_v_range[1] - q_v_range[0]) / bins[1],
                                              bins=bins,
                                              pos0_range=q_h_range,
                                              pos1_range=q_v_range,
                                              dummy=None,
                                              delta_dummy=None,
                                              allow_pos0_neg=True,
                                              mask=mask,
                                              #dark=dark,
                                              #flat=flat,
                                              #solidangle=solidangle,
                                              #polarization=polarization,
                                              #normalization_factor=normalization_factor,
                                              chiDiscAtPi=1,
                                              )

    return I, q_y, q_z, resc_q


def q_from_angles(phi, alpha, wavelength):
    """
    Conversion of angle to q values for SAXS configuration

    Parameters:
    -----------
    :param phi: 2D array containing the radial angle of each pixel in the image
    :type phi: ndarray
    :param alpha: 2D array containing the azimuthal angle of each pixel in the image
    :type alpha: ndarray
    :param wavelength: wavelength of the x-rays
    :type wavelength: float
    """
    r = 4 * np.pi / wavelength
    qx = r * np.sin(0.5*phi) * np.cos(0.5*alpha)
    qy = r * np.sin(0.5*alpha)
    qz = r * np.cos(0.5*alpha) * np.cos(0.5*alpha) - 1
    return np.array([qx, qy, qz])


def alpha(x, y, z):
    """
    Conversion each pixel of the image in azimuthal angle

    Parameters:
    -----------
    :param x: 2D array containing the X of each pixel in the image
    :type x: ndarray
    :param y: 2D array containing the Y of each pixel in the image
    :type y: ndarray
    :param z: 2D array containing the Z of each pixel in the image
    :type z: ndarray
    """
    return np.arctan2(y, np.sqrt(x ** 2 + z ** 2))


def phi(x, y, z):
    """
    Conversion each pixel of the image in radial angle

    Parameters:
    -----------
    :param x: 2D array containing the X of each pixel in the image
    :type x: ndarray
    :param y: 2D array containing the Y of each pixel in the image
    :type y: ndarray
    :param z: 2D array containing the Z of each pixel in the image
    :type z: ndarray
    """
    return np.arctan2(x, np.sqrt(z ** 2))

# %%