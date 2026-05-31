
    def tick_values(self, vmin, vmax):
        """
        Return the values of the located ticks given **vmin** and **vmax**.

        .. note::
            To get tick locations with the vmin and vmax values defined
            automatically for the associated :attr:`axis` simply call
            the Locator instance::

                >>> print(type(loc))
                <type 'Locator'>
                >>> print(loc())
                [1, 2, 3, 4]

        """
        raise NotImplementedError('Derived must override')

    def set_params(self, **kwargs):
        """
        Do nothing, and raise a warning. Any locator class not supporting the
        set_params() function will call this.
        """
        cbook._warn_external(
            "'set_params()' not defined for locator of type " +
            str(type(self)))

    def __call__(self):
        """Return the locations of the ticks"""
        # note: some locators return data limits, other return view limits,
        # hence there is no *one* interface to call self.tick_values.
        raise NotImplementedError('Derived must override')

    def raise_if_exceeds(self, locs):
        """raise a RuntimeError if Locator attempts to create more than
           MAXTICKS locs"""
        if len(locs) >= self.MAXTICKS:
            raise RuntimeError("Locator attempting to generate {} ticks from "
                               "{} to {}: exceeds Locator.MAXTICKS".format(
                                   len(locs), locs[0], locs[-1]))
        return locs

    def nonsingular(self, v0, v1):
        """Modify the endpoints of a range as needed to avoid singularities."""
        return mtransforms.nonsingular(v0, v1, increasing=False, expander=.05)

    def view_limits(self, vmin, vmax):
        """
        Select a scale for the range from vmin to vmax.

        Subclasses should override this method to change locator behaviour.
        """
        return mtransforms.nonsingular(vmin, vmax)

    def autoscale(self):
        """autoscale the view limits"""
        return self.view_limits(*self.axis.get_view_interval())

    def pan(self, numsteps):
        """Pan numticks (can be positive or negative)"""
        ticks = self()
        numticks = len(ticks)

        vmin, vmax = self.axis.get_view_interval()
        vmin, vmax = mtransforms.nonsingular(vmin, vmax, expander=0.05)
        if numticks > 2:
            step = numsteps * abs(ticks[0] - ticks[1])
        else:
            d = abs(vmax - vmin)
            step = numsteps * d / 6.

        vmin += step
        vmax += step
        self.axis.set_view_interval(vmin, vmax, ignore=True)

    def zoom(self, direction):
        "Zoom in/out on axis; if direction is >0 zoom in, else zoom out"

        vmin, vmax = self.axis.get_view_interval()
        vmin, vmax = mtransforms.nonsingular(vmin, vmax, expander=0.05)
        interval = abs(vmax - vmin)
        step = 0.1 * interval * direction
        self.axis.set_view_interval(vmin + step, vmax - step, ignore=True)

    def refresh(self):
        """refresh internal information based on current lim"""
        pass
