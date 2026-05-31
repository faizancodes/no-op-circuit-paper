        # The implementation can move back into clear() when the
        # deprecation on cla() subclassing expires.

        # stash the current visibility state
        if hasattr(self, 'patch'):
            patch_visible = self.patch.get_visible()
        else:
            patch_visible = True

        xaxis_visible = self.xaxis.get_visible()
        yaxis_visible = self.yaxis.get_visible()

        for axis in self._axis_map.values():
            axis.clear()  # Also resets the scale to linear.
        for spine in self.spines.values():
            spine.clear()

        self.ignore_existing_data_limits = True
        self.callbacks = cbook.CallbackRegistry(
            signals=["xlim_changed", "ylim_changed", "zlim_changed"])

        # update the minor locator for x and y axis based on rcParams
        if mpl.rcParams['xtick.minor.visible']:
            self.xaxis.set_minor_locator(mticker.AutoMinorLocator())
        if mpl.rcParams['ytick.minor.visible']:
            self.yaxis.set_minor_locator(mticker.AutoMinorLocator())

        self._xmargin = mpl.rcParams['axes.xmargin']
        self._ymargin = mpl.rcParams['axes.ymargin']
        self._tight = None
        self._use_sticky_edges = True

        self._get_lines = _process_plot_var_args(self)
        self._get_patches_for_fill = _process_plot_var_args(self, 'fill')

        self._gridOn = mpl.rcParams['axes.grid']
        self._children = []
        self._mouseover_set = _OrderedSet()
        self.child_axes = []
        self._current_image = None  # strictly for pyplot via _sci, _gci
        self._get_patches_for_fill = _process_plot_var_args(self, 'fill')

        self._gridOn = mpl.rcParams['axes.grid']
        old_children, self._children = self._children, []
        for chld in old_children:
            chld.axes = chld.figure = None
        self._mouseover_set = _OrderedSet()
        self.child_axes = []
        self._current_image = None  # strictly for pyplot via _sci, _gci
        props = font_manager.FontProperties(
            size=mpl.rcParams['axes.titlesize'],
            weight=mpl.rcParams['axes.titleweight'])

        y = mpl.rcParams['axes.titley']
        if y is None:
            y = 1.0
            self._autotitlepos = True
        else:
            self._autotitlepos = False

        self.title = mtext.Text(
            x=0.5, y=y, text='',
            fontproperties=props,
            verticalalignment='baseline',
            horizontalalignment='center',
            )
        self._left_title = mtext.Text(
            x=0.0, y=y, text='',
            fontproperties=props.copy(),
            verticalalignment='baseline',
            horizontalalignment='left', )
        self._right_title = mtext.Text(
            x=1.0, y=y, text='',
            fontproperties=props.copy(),
            verticalalignment='baseline',
            horizontalalignment='right',
            )
        title_offset_points = mpl.rcParams['axes.titlepad']
        # refactor this out so it can be called in ax.set_title if
        # pad argument used...
        self._set_title_offset_trans(title_offset_points)

        for _title in (self.title, self._left_title, self._right_title):
            self._set_artist_props(_title)

        # The patch draws the background of the Axes.  We want this to be below
        # the other artists.  We use the frame to draw the edges so we are
        # setting the edgecolor to None.
        self.patch = self._gen_axes_patch()
