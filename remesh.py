# %%


# %%
import numpy as np
from pyFAI.ext import splitBBox


def remesh_gi(data, ai, npt=None, q_h_range=None, q_v_range=None, method='splitpix',
              mask=None, incident_angle=0.0, tilt_angle=0.0, sample_orientation=7):
    """
    Redraw the Grazing-Incidence image in (q_par, q_ver) coordinates using
    pyFAI FiberIntegrator.integrate2d_grazing_incidence().

    For sample_orientation=7 the FiberIntegrator returns:
        raw q_ip  : -0.305 → +0.099  (physically -q_ver, out-of-plane but negated)
        raw q_oop : -0.077 → +0.278  (physically  q_par, in-plane, correct sign)
        intensity : shape (npt_ip, npt_oop) = (q_z rows, q_par cols)

    Corrections applied before returning:
        q_par =  q_oop          (in-plane,     keep as-is)
        q_ver = -q_ip           (out-of-plane, negate to get -0.099 → +0.305)
        intensity = flipud       (vertical flip to match negated q_ver axis)

    No transpose is needed — the intensity rows already correspond to q_ver
    and columns already correspond to q_par after the vertical flip.

    q_h_range and q_v_range are interpreted in the corrected axis convention
    (q_par and q_ver respectively) and converted internally to the raw
    FiberIntegrator ip_range/oop_range convention.

    Parameters:
    -----------
    :param data: 2D image in pixel
    :param ai: pyFAI FiberIntegrator (promoted from AzimuthalIntegrator)
    :param npt: number of points. Tuple (npt_qp, npt_qz), single int, or None.
    :param q_h_range: (min, max) in-plane q_par range in A^-1
    :param q_v_range: (min, max) out-of-plane q_ver range in A^-1
    :param method: pyFAI integration method string
    :param mask: mask array same shape as data (1=masked, 0=valid)
    :param incident_angle: grazing incident angle in radians
    :param tilt_angle: tilt angle in radians
    :param sample_orientation: FiberIntegrator sample orientation (1-8), default 7
    """

    if npt is None:
        # intensity shape is (npt_ip rows, npt_oop cols).
        # rows = q_ver = detector height = data.shape[0]
        # cols = q_par = detector width  = data.shape[1]
        npt_ip  = data.shape[0]
        npt_oop = data.shape[1]
    elif isinstance(npt, (tuple, list)):
        # npt arrives from stitch.py as (npt_qp, npt_qz).
        # rows = npt_ip = npt_qz = npt[1]
        # cols = npt_oop = npt_qp = npt[0]
        npt_ip  = int(npt[1])
        npt_oop = int(npt[0])
    else:
        npt_ip  = int(npt)
        npt_oop = int(npt)

    # q_h_range is in the corrected q_par convention where q_par = q_oop.
    # Pass directly as oop_range — no conversion needed.
    if q_h_range is not None:
        oop_range = (q_h_range[0], q_h_range[1])
    else:
        oop_range = None

    # q_v_range is in the corrected q_ver convention where q_ver = -q_ip.
    # Convert to raw ip_range:
    #   q_ver = -q_ip  →  q_ip = -q_ver
    #   ip_range = (-q_v_range[1], -q_v_range[0])  (note the reversal of min/max)
    if q_v_range is not None:
        ip_range = (-q_v_range[1], -q_v_range[0])
    else:
        ip_range = None

    result = ai.integrate2d_grazing_incidence(
        data=data,
        npt_ip=npt_ip,
        npt_oop=npt_oop,
        sample_orientation=sample_orientation,
        incident_angle=incident_angle,
        tilt_angle=tilt_angle,
        unit_ip='qip_A^-1',
        unit_oop='qoop_A^-1',
        ip_range=ip_range,
        oop_range=oop_range,
        method=method,
        mask=mask,
    )

    intensity, q_ip, q_oop = result

    # Apply axis correction:
    #   q_par (in-plane)     =  q_oop  (keep as-is)
    #   q_ver (out-of-plane) = -q_ip   (negate → positive upward)
    q_par =  q_oop
    q_ver = -q_ip

    # intensity shape is (npt_ip, npt_oop) = (q_ver rows, q_par cols).
    # flipud because q_ver = -q_ip reverses the row order.
    # No transpose needed — axes are already in the correct orientation.
    intensity_out = np.flipud(intensity)

    return intensity_out, q_par, q_ver



# def remesh_gi(data, ai, npt=None, q_h_range=None, q_v_range=None, method='bbox' , mask=None,                 #method='bbox'
#               incident_angle=0.0, tilt_angle=0.0, sample_orientation=4):
#     """
#     Redraw the Grazing-Incidence image in (qp, qz) coordinates using pyGIX

#     Parameters:
#     -----------
#     :param data: 2D image in pixel
#     :type data: numpy 2D array of float
#     :param ai: pyGIX transform operator
#     :type ai: pyGIXTransform operator
#     :param npt: number of point for the binning
#     :type npt: int
#     :param q_h_range: Starting and ending point for the q_horizontal range
#     :type q_h_range: Tuple(float, float), optional
#     :param q_v_range: Starting and ending point for the q_vertical range
#     :type q_v_range: Tuple(float, float), optional
#     :param method: Method for the remeshing
#     :type method: String: 'splitbbox', ...
#     :param mask: Mask of the 2D raw image
#     :type mask: numpy 2D array of boolean
#     :param incident_angle: kept for API compatibility with stitch.py and xray_analysis.py
#     :type incident_angle: float
#     :param tilt_angle: kept for API compatibility with stitch.py and xray_analysis.py
#     :type tilt_angle: float
#     :param sample_orientation: kept for API compatibility with stitch.py and xray_analysis.py
#     :type sample_orientation: int
#     """

#     img, q_par, q_ver = ai.transform_reciprocal(data,
#                                                 npt=npt,
#                                                 ip_range=q_h_range,
#                                                 op_range=q_v_range,
#                                                 method=method,
#                                                 unit='A',
#                                                 mask=mask)

#     return img, q_par, q_ver


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
