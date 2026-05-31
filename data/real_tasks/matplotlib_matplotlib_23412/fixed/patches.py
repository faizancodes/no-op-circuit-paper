        gc.set_foreground(self._edgecolor, isRGBA=True)

        lw = self._linewidth
        if self._edgecolor[3] == 0 or self._linestyle == 'None':
            lw = 0
        gc.set_linewidth(lw)
        gc.set_dashes(*self._dash_pattern)
        gc.set_capstyle(self._capstyle)
        gc.set_joinstyle(self._joinstyle)

        gc.set_antialiased(self._antialiased)
        self._set_gc_clip(gc)
        gc.set_url(self._url)
        gc.set_snap(self.get_snap())

        gc.set_alpha(self._alpha)

        if self._hatch:
            gc.set_hatch(self._hatch)
            gc.set_hatch_color(self._hatch_color)

        if self.get_sketch_params() is not None:
            gc.set_sketch_params(*self.get_sketch_params())

        if self.get_path_effects():
            from matplotlib.patheffects import PathEffectRenderer
            renderer = PathEffectRenderer(self.get_path_effects(), renderer)

        # In `with _bind_draw_path_function(renderer) as draw_path: ...`
        # (in the implementations of `draw()` below), calls to `draw_path(...)`
        # will occur as if they took place here with `gc` inserted as
        # additional first argument.
        yield functools.partial(renderer.draw_path, gc)

        gc.restore()
        renderer.close_group('patch')
        self.stale = False

    @artist.allow_rasterization
    def draw(self, renderer):
        # docstring inherited
        if not self.get_visible():
            return

        with cbook._setattr_cm(self, _dash_pattern=(self._dash_pattern)), \
             self._bind_draw_path_function(renderer) as draw_path:
            path = self.get_path()
            transform = self.get_transform()
            tpath = transform.transform_path_non_affine(path)
            affine = transform.get_affine()
            draw_path(tpath, affine,
                      # Work around a bug in the PDF and SVG renderers, which
                      # do not draw the hatches if the facecolor is fully
                      # transparent, but do if it is None.
                      self._facecolor if self._facecolor[3] else None)

    def get_path(self):
        """Return the path of this patch."""
        raise NotImplementedError('Derived must override')

    def get_window_extent(self, renderer=None):
        return self.get_path().get_extents(self.get_transform())

    def _convert_xy_units(self, xy):
        """Convert x and y units for a tuple (x, y)."""
        x = self.convert_xunits(xy[0])
        y = self.convert_yunits(xy[1])
        return x, y


class Shadow(Patch):
    def __str__(self):
        return "Shadow(%s)" % (str(self.patch))

    @_docstring.dedent_interpd
    def __init__(self, patch, ox, oy, **kwargs):
        """
        Create a shadow of the given *patch*.

        By default, the shadow will have the same face color as the *patch*,
        but darkened.

        Parameters
        ----------
        patch : `.Patch`
            The patch to create the shadow for.
        ox, oy : float
            The shift of the shadow in data coordinates, scaled by a factor
