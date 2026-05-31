        # In the x-direction, the hexagons exactly cover the region from
        # xmin to xmax. Need some padding to avoid roundoff errors.
        padding = 1.e-9 * (xmax - xmin)
        xmin -= padding
        xmax += padding
        sx = (xmax - xmin) / nx
        sy = (ymax - ymin) / ny
        # Positions in hexagon index coordinates.
        ix = (tx - xmin) / sx
        iy = (ty - ymin) / sy
        ix1 = np.round(ix).astype(int)
        iy1 = np.round(iy).astype(int)
        ix2 = np.floor(ix).astype(int)
        iy2 = np.floor(iy).astype(int)
        # flat indices, plus one so that out-of-range points go to position 0.
        i1 = np.where((0 <= ix1) & (ix1 < nx1) & (0 <= iy1) & (iy1 < ny1),
                      ix1 * ny1 + iy1 + 1, 0)
        i2 = np.where((0 <= ix2) & (ix2 < nx2) & (0 <= iy2) & (iy2 < ny2),
                      ix2 * ny2 + iy2 + 1, 0)

        d1 = (ix - ix1) ** 2 + 3.0 * (iy - iy1) ** 2
        d2 = (ix - ix2 - 0.5) ** 2 + 3.0 * (iy - iy2 - 0.5) ** 2
        bdist = (d1 < d2)

        if C is None:  # [1:] drops out-of-range points.
            counts1 = np.bincount(i1[bdist], minlength=1 + nx1 * ny1)[1:]
            counts2 = np.bincount(i2[~bdist], minlength=1 + nx2 * ny2)[1:]
            accum = np.concatenate([counts1, counts2]).astype(float)
            if mincnt is not None:
                accum[accum < mincnt] = np.nan
            C = np.ones(len(x))
        else:
            # store the C values in a list per hexagon index
            Cs_at_i1 = [[] for _ in range(1 + nx1 * ny1)]
            Cs_at_i2 = [[] for _ in range(1 + nx2 * ny2)]
            for i in range(len(x)):
                if bdist[i]:
                    Cs_at_i1[i1[i]].append(C[i])
                else:
                    Cs_at_i2[i2[i]].append(C[i])
            if mincnt is None:
                mincnt = 0
            accum = np.array(
                [reduce_C_function(acc) if len(acc) > mincnt else np.nan
                 for Cs_at_i in [Cs_at_i1, Cs_at_i2]
                 for acc in Cs_at_i[1:]],  # [1:] drops out-of-range points.
                float)

        good_idxs = ~np.isnan(accum)

        offsets = np.zeros((n, 2), float)
        offsets[:nx1 * ny1, 0] = np.repeat(np.arange(nx1), ny1)
        offsets[:nx1 * ny1, 1] = np.tile(np.arange(ny1), nx1)
        offsets[nx1 * ny1:, 0] = np.repeat(np.arange(nx2) + 0.5, ny2)
        offsets[nx1 * ny1:, 1] = np.tile(np.arange(ny2), nx2) + 0.5
        offsets[:, 0] *= sx
        offsets[:, 1] *= sy
        offsets[:, 0] += xmin
        offsets[:, 1] += ymin
        # remove accumulation bins with no data
        offsets = offsets[good_idxs, :]
        accum = accum[good_idxs]

        polygon = [sx, sy / 3] * np.array(
            [[.5, -.5], [.5, .5], [0., 1.], [-.5, .5], [-.5, -.5], [0., -1.]])

        if linewidths is None:
            linewidths = [mpl.rcParams['patch.linewidth']]

        if xscale == 'log' or yscale == 'log':
            polygons = np.expand_dims(polygon, 0) + np.expand_dims(offsets, 1)
            if xscale == 'log':
                polygons[:, :, 0] = 10.0 ** polygons[:, :, 0]
                xmin = 10.0 ** xmin
                xmax = 10.0 ** xmax
                self.set_xscale(xscale)
            if yscale == 'log':
                polygons[:, :, 1] = 10.0 ** polygons[:, :, 1]
                ymin = 10.0 ** ymin
                ymax = 10.0 ** ymax
                self.set_yscale(yscale)
            collection = mcoll.PolyCollection(
                polygons,
                edgecolors=edgecolors,
                linewidths=linewidths,
                )
        else:
