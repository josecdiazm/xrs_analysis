# %%


# %%

import numpy as np
import remesh


def stitching(datas, ais, masks, geometry='Reflection', interp_factor=2,
              flag_scale=True, resc_q=False, _cached_qranges=None):
    '''
    Remeshing in q-space the 2D image collected by the pixel detector and stitching together
    images at different detector position (if several images).

    Parameters:
    -----------
    :param datas: List of 2D images in pixel
    :param ais: List of AzimuthalIntegrator/Transform objects
    :param masks: List of masks
    :param geometry: 'Reflection' or 'Transmission'
    :param interp_factor: factor to increase the binning of the stitched image
    :param flag_scale: Boolean to scale consecutive images at different detector positions
    :param resc_q: Rescale qs (fix for pyFAI bug when qs exceed pi)
    :param _cached_qranges: dict with keys 'qp_remesh', 'qz_remesh', and 'q_p_ini', 'q_z_ini'
                            from a previous call with the same geometry/detector configuration.
                            If provided, the expensive scout remesh pass is skipped entirely.
    '''

    if _cached_qranges is None:
        # -----------------------------------------------------------------------
        # SCOUT PASS: run remesh once per image just to find the q-space extent
        # and the per-image q-ranges needed for stitching overlap calculation.
        # This only runs on the very first call for a given geometry setup.
        # For fixed geometry (same detector, SDD, beam center, wavelength, alphai)
        # the result is identical every time, so we cache and skip it after.
        # -----------------------------------------------------------------------
        for i, (data, ai, mask) in enumerate(zip(datas, ais, masks)):
            if geometry == 'Reflection':
                img, x, y = remesh.remesh_gi(data, ai, method='splitbbox', mask=mask)
                if i == 0:
                    q_p_ini = np.zeros((np.shape(x)[0], len(datas)))
                    q_z_ini = np.zeros((np.shape(y)[0], len(datas)))
                q_p_ini[:len(x), i] = -x[::-1]
                q_z_ini[:len(y), i] = y[::-1]

            elif geometry == 'Transmission':
                img, x, y, resc_q = remesh.remesh_transmission(data, ai, mask=mask)
                if i == 0:
                    q_p_ini = np.zeros((np.shape(x)[0], len(datas)))
                    q_z_ini = np.zeros((np.shape(y)[0], len(datas)))
                q_p_ini[:len(x), i] = x
                q_z_ini[:len(y), i] = y

        nb_point = len(q_p_ini[:, 0])
        for i in range(1, np.shape(q_p_ini)[1], 1):
            y_idx = np.argmin(abs(q_p_ini[:, i - 1] - np.min(q_p_ini[:, i])))
            nb_point += len(q_p_ini[:, i]) - y_idx

        nb_point = nb_point * interp_factor
        qp_remesh = np.linspace(min(q_p_ini[:, 0]), max(q_p_ini[:, -1]), nb_point)
        qz_remesh = np.linspace(
            min(q_z_ini[:, 0]), max(q_z_ini[:, -1]),
            int(nb_point * abs(max(q_z_ini[:, -1]) - min(q_z_ini[:, 0])) /
                abs(max(q_p_ini[:, -1]) - min(q_p_ini[:, 0])))
        )

    else:
        # -----------------------------------------------------------------------
        # CACHED PATH: geometry hasn't changed, reuse everything from last time.
        # This saves the entire scout remesh pass (~half of total stitching time).
        # q_p_ini and q_z_ini are also cached because they are needed to compute
        # the per-image qp_start/qp_stop overlap positions during stitching,
        # and those positions depend only on the detector angles, not the data.
        # -----------------------------------------------------------------------
        qp_remesh = _cached_qranges['qp_remesh']
        qz_remesh = _cached_qranges['qz_remesh']
        q_p_ini   = _cached_qranges['q_p_ini']
        q_z_ini   = _cached_qranges['q_z_ini']

    # ---------------------------------------------------------------------------
    # MAIN PASS: remesh each image onto the target q-grid and stitch.
    # This part always runs because it depends on the image data.
    # ---------------------------------------------------------------------------
    for i, (data, ai, mask) in enumerate(zip(datas, ais, masks)):

        # Compute where this image sits on the shared q-grid.
        # For a single image this is always the full range.
        # For multi-angle stitching each image occupies a different sub-range.
        qp_start = np.argmin(abs(qp_remesh - np.min(q_p_ini[:, i])))
        qp_stop  = np.argmin(abs(qp_remesh - np.max(q_p_ini[:, i])))
        npt = (int(qp_stop - qp_start), int(np.shape(qz_remesh)[0]))

        if geometry == 'Reflection':
            ip_range = (-qp_remesh[qp_stop], -qp_remesh[qp_start])
            op_range = (qz_remesh[0], qz_remesh[-1])
            msk, _, _ = remesh.remesh_gi(mask.astype(int), ai, npt=npt,
                                         q_h_range=ip_range, q_v_range=op_range,
                                         method='splitbbox', mask=mask)
            img, x, y = remesh.remesh_gi(data, ai, npt=npt,
                                         q_h_range=ip_range, q_v_range=op_range,
                                         method='splitbbox', mask=mask)
            qimage, qmask = np.rot90(img, 2), np.rot90(msk, 2)
            qp, qz = -x[::-1], y[::-1]

        elif geometry == 'Transmission':
            ip_range = (qp_remesh[qp_start], qp_remesh[qp_stop])
            op_range = (qz_remesh[0], qz_remesh[-1])
            qmask, _, _, _ = remesh.remesh_transmission(mask.astype(int), ai, bins=npt,
                                                        q_h_range=ip_range, q_v_range=op_range,
                                                        mask=None)
            qimage, x, y, resc_q = remesh.remesh_transmission(data, ai, bins=npt,
                                                               q_h_range=ip_range, q_v_range=op_range,
                                                               mask=mask)

        if i == 0:
            img_te   = np.zeros((np.shape(qz_remesh)[0], np.shape(qp_remesh)[0]))
            img_mask = np.zeros(np.shape(img_te))
            sca  = np.zeros(np.shape(img_te))
            sca1 = np.zeros(np.shape(img_te))
            sca2 = np.zeros(np.shape(img_te))
            sca3 = np.zeros(np.shape(img_te))
            img_te[:, :np.shape(qimage)[1]] = qimage
            img_mask[:, :np.shape(qmask)[1]] += np.logical_not(qmask).astype(int)
            sca[np.nonzero(qimage)]  += 1
            sca2[np.nonzero(qimage)] += 1
            sca3[np.nonzero(qimage)] += 1
            scale  = 1.
            scales = [scale]

        else:
            if flag_scale:
                threshold = 0.01
            else:
                threshold = 0.000001

            sca1 = np.ones(np.shape(sca)) * sca
            sca1[:, qp_start: qp_start + np.shape(qimage)[1]] += (qimage >= threshold).astype(int)

            img1 = np.ma.masked_array(img_te, mask=sca1 != 2 * sca)
            img1 = np.ma.masked_where(img1 < threshold, img1)
            img_te[:, qp_start:qp_start + np.shape(qimage)[1]] += qimage
            img_mask[:, qp_start:qp_start + np.shape(qimage)[1]] += (
                (qimage >= threshold).astype(int) * np.logical_not(qmask).astype(int))

            img2 = np.ma.masked_array(img_te, mask=sca1 != 2 * sca)
            img2 = np.ma.masked_where(img2 < threshold, img2)

            scale *= abs(np.mean(img2) - np.mean(img1)) / np.mean(img1)
            if np.ma.is_masked(scale):
                scale = scales[i - 1]

            sca[:, qp_start: qp_start + np.shape(qimage)[1]] += (qimage >= threshold).astype(int)

            if flag_scale:
                if ai.detector.aliases[0] != 'Pilatus 900kw (Vertical)':
                    sca2[:, qp_start: qp_start + np.shape(qimage)[1]] += (qimage >= threshold).astype(int) * scale
                    scales.append(scale)
                else:
                    if i % 3 == 0:
                        sca3 = np.zeros(np.shape(img_te))
                        sca3[:, qp_start: qp_start + np.shape(qimage)[1]] += (qimage >= threshold).astype(int)
                        scales.append(scale)
                    elif i % 3 == 1:
                        sca4 = np.zeros(np.shape(img_te))
                        sca4[:, qp_start: qp_start + np.shape(qimage)[1]] += (qimage >= threshold).astype(int)
                        scales.append(scale)
                    elif i % 3 == 2:
                        sca5 = np.zeros(np.shape(img_te))
                        sca5[:, qp_start: qp_start + np.shape(qimage)[1]] += (qimage >= threshold).astype(int)
                        scales.append(scale)
                        scale_max = np.max(scales[i - 2:i])
                        if i == 2:
                            sca2 = np.zeros(np.shape(img_te))
                        sca2 += sca3 * scale_max
                        sca2 += sca4 * scale_max
                        sca2 += sca5 * scale_max
                        scales[i - 2], scales[i - 1], scales[i] = scale_max, scale_max, scale_max
            else:
                sca2[:, qp_start: qp_start + np.shape(qimage)[1]] += (qimage >= threshold).astype(int)
                scales.append(1)

    sca2[np.where(sca2 == 0)] = 1
    img = img_te / sca2
    mask_out = (img_mask.astype(bool)).astype(int)

    if geometry == 'Reflection':
        img = np.flipud(img)

    qp_out = [qp_remesh.min(), qp_remesh.max()]
    qz_out = [-qz_remesh.max(), -qz_remesh.min()]

    if resc_q:
        qp_out[:] = [v * 10 for v in qp_out]
        qz_out[:] = [v * 10 for v in qz_out]

    # Bundle everything the caller needs to skip the scout pass next time.
    # q_p_ini and q_z_ini are included because they encode the per-image
    # q-space positions which depend on detector angles, not image data.
    cached_qranges = {
        'qp_remesh': qp_remesh,
        'qz_remesh': qz_remesh,
        'q_p_ini':   q_p_ini,
        'q_z_ini':   q_z_ini,
    }

    return img, mask_out, qp_out, qz_out, scales, cached_qranges


def translation_stitching(datas, masks=None, pys=None, pxs=None):
    if not pxs:
        pxs = [0] * len(datas)
    if not pys:
        pys = [0] * len(datas)
    if not masks:
            masks = [np.zeros((np.shape(datas[0])))] * len(datas)

    for i, (data, mask, py, px) in enumerate(zip(datas, masks, pys, pxs)):
        delta_y = py - np.min(pys)
        delta_y1 = py - np.max(pys)
        padtop = int(abs(round(delta_y/0.172)))
        padbottom = int(abs(round(delta_y1/0.172)))

        delta_x = px - np.min(pxs)
        delta_x1 = px - np.max(pxs)
        padleft = int(abs(round(delta_x/0.172)))
        padright = int(abs(round(delta_x1/0.172)))

        d = np.pad(data, ((padtop, padbottom), (padleft, padright)), 'constant')
        d[np.where(d < 0)] = 0
        m = np.pad(1 - mask, ((padtop, padbottom), (padleft, padright)), 'constant')

        if i == 0:
            dat_sum = d
            mask_sum = m

        else:
            dat_sum = dat_sum + d
            mask_sum = mask_sum + m

    data = dat_sum/mask_sum
    data[np.isnan(data)] = 0
    return data


# %%



