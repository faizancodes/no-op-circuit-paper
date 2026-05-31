        else:
            color = mcolors.to_rgba_array(color)
            if len(color) != nx:
                error_message = (
                    "color kwarg must have one color per data set. %d data "
                    "sets and %d colors were provided" % (nx, len(color)))
                raise ValueError(error_message)

        hist_kwargs = dict()

        # if the bin_range is not given, compute without nan numpy
        # does not do this for us when guessing the range (but will
        # happily ignore nans when computing the histogram).
        if bin_range is None:
            xmin = np.inf
            xmax = -np.inf
            for xi in x:
                if len(xi):
                    # python's min/max ignore nan,
                    # np.minnan returns nan for all nan input
                    xmin = min(xmin, np.nanmin(xi))
                    xmax = max(xmax, np.nanmax(xi))
            # make sure we have seen at least one non-nan and finite
            # value before we reset the bin range
            if not np.isnan([xmin, xmax]).any() and not (xmin > xmax):
                bin_range = (xmin, xmax)

        # If bins are not specified either explicitly or via range,
        # we need to figure out the range required for all datasets,
        # and supply that to np.histogram.
        if not input_empty and len(x) > 1:
            if weights is not None:
                _w = np.concatenate(w)
            else:
                _w = None

            bins = histogram_bin_edges(np.concatenate(x),
                                       bins, bin_range, _w)
        else:
            hist_kwargs['range'] = bin_range

        density = bool(density) or bool(normed)
        if density and not stacked:
            hist_kwargs = dict(density=density)

        # List to store all the top coordinates of the histograms
        tops = []
        mlast = None
        # Loop through datasets
        for i in range(nx):
            # this will automatically overwrite bins,
            # so that each histogram uses the same bins
            m, bins = np.histogram(x[i], bins, weights=w[i], **hist_kwargs)
            m = m.astype(float)  # causes problems later if it's an int
            if mlast is None:
                mlast = np.zeros(len(bins)-1, m.dtype)
            if stacked:
                m += mlast
                mlast[:] = m
            tops.append(m)

        # If a stacked density plot, normalize so the area of all the stacked
        # histograms together is 1
        if stacked and density:
            db = np.diff(bins)
            for m in tops:
                m[:] = (m / db) / tops[-1].sum()
        if cumulative:
            slc = slice(None)
            if isinstance(cumulative, Number) and cumulative < 0:
                slc = slice(None, None, -1)

            if density:
                tops = [(m * np.diff(bins))[slc].cumsum()[slc] for m in tops]
            else:
                tops = [m[slc].cumsum()[slc] for m in tops]

        patches = []

        # Save autoscale state for later restoration; turn autoscaling
        # off so we can do it all a single time at the end, instead
        # of having it done by bar or fill and then having to be redone.
        _saved_autoscalex = self.get_autoscalex_on()
        _saved_autoscaley = self.get_autoscaley_on()
        self.set_autoscalex_on(False)
        self.set_autoscaley_on(False)
