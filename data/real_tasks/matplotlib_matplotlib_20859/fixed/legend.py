                value = mpl.rcParams["legend." + name]
            else:
                value = locals_view[name]
            setattr(self, name, value)
        del locals_view
        # trim handles and labels if illegal label...
        _lab, _hand = [], []
        for label, handle in zip(labels, handles):
            if isinstance(label, str) and label.startswith('_'):
                _api.warn_external('The handle {!r} has a label of {!r} '
                                   'which cannot be automatically added to'
                                   ' the legend.'.format(handle, label))
            else:
                _lab.append(label)
                _hand.append(handle)
        labels, handles = _lab, _hand

        handles = list(handles)
        if len(handles) < 2:
            ncol = 1
        self._ncol = ncol

        if self.numpoints <= 0:
            raise ValueError("numpoints must be > 0; it was %d" % numpoints)

        # introduce y-offset for handles of the scatter plot
        if scatteryoffsets is None:
            self._scatteryoffsets = np.array([3. / 8., 4. / 8., 2.5 / 8.])
        else:
            self._scatteryoffsets = np.asarray(scatteryoffsets)
        reps = self.scatterpoints // len(self._scatteryoffsets) + 1
        self._scatteryoffsets = np.tile(self._scatteryoffsets,
                                        reps)[:self.scatterpoints]

        # _legend_box is a VPacker instance that contains all
        # legend items and will be initialized from _init_legend_box()
        # method.
        self._legend_box = None

        if isinstance(parent, Axes):
            self.isaxes = True
            self.axes = parent
            self.set_figure(parent.figure)
        elif isinstance(parent, FigureBase):
            self.isaxes = False
            self.set_figure(parent)
        else:
            raise TypeError(
                "Legend needs either Axes or FigureBase as parent"
            )
        self.parent = parent

        self._loc_used_default = loc is None
        if loc is None:
            loc = mpl.rcParams["legend.loc"]
            if not self.isaxes and loc in [0, 'best']:
                loc = 'upper right'
        if isinstance(loc, str):
            loc = _api.check_getitem(self.codes, loc=loc)
        if not self.isaxes and loc == 0:
            raise ValueError(
                "Automatic legend placement (loc='best') not implemented for "
                "figure legend")

        self._mode = mode
        self.set_bbox_to_anchor(bbox_to_anchor, bbox_transform)

        # We use FancyBboxPatch to draw a legend frame. The location
        # and size of the box will be updated during the drawing time.

        if facecolor is None:
            facecolor = mpl.rcParams["legend.facecolor"]
        if facecolor == 'inherit':
            facecolor = mpl.rcParams["axes.facecolor"]

        if edgecolor is None:
            edgecolor = mpl.rcParams["legend.edgecolor"]
        if edgecolor == 'inherit':
            edgecolor = mpl.rcParams["axes.edgecolor"]

        if fancybox is None:
            fancybox = mpl.rcParams["legend.fancybox"]

        self.legendPatch = FancyBboxPatch(
            xy=(0, 0), width=1, height=1,
            facecolor=facecolor, edgecolor=edgecolor,
            # If shadow is used, default to alpha=1 (#8943).
            alpha=(framealpha if framealpha is not None
                   else 1 if shadow
                   else mpl.rcParams["legend.framealpha"]),
            # The width and height of the legendPatch will be set (in draw())
            # to the length that includes the padding. Thus we set pad=0 here.
            boxstyle=("round,pad=0,rounding_size=0.2" if fancybox
