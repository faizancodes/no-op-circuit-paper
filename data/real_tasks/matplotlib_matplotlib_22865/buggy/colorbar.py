        lower, upper = self.vmin, self.vmax
        if self._long_axis().get_inverted():
            # If the axis is inverted, we need to swap the vmin/vmax
            lower, upper = upper, lower
        if self.orientation == 'vertical':
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(lower, upper)
        else:
            self.ax.set_ylim(0, 1)
            self.ax.set_xlim(lower, upper)

        # set up the tick locators and formatters.  A bit complicated because
        # boundary norms + uniform spacing requires a manual locator.
        self.update_ticks()

        if self._filled:
            ind = np.arange(len(self._values))
            if self._extend_lower():
                ind = ind[1:]
            if self._extend_upper():
                ind = ind[:-1]
            self._add_solids(X, Y, self._values[ind, np.newaxis])

    def _add_solids(self, X, Y, C):
        """Draw the colors; optionally add separators."""
        # Cleanup previously set artists.
        if self.solids is not None:
            self.solids.remove()
        for solid in self.solids_patches:
            solid.remove()
        # Add new artist(s), based on mappable type.  Use individual patches if
        # hatching is needed, pcolormesh otherwise.
        mappable = getattr(self, 'mappable', None)
        if (isinstance(mappable, contour.ContourSet)
                and any(hatch is not None for hatch in mappable.hatches)):
            self._add_solids_patches(X, Y, C, mappable)
        else:
            self.solids = self.ax.pcolormesh(
                X, Y, C, cmap=self.cmap, norm=self.norm, alpha=self.alpha,
                edgecolors='none', shading='flat')
            if not self.drawedges:
                if len(self._y) >= self.n_rasterize:
                    self.solids.set_rasterized(True)
        self.dividers.set_segments(
            np.dstack([X, Y])[1:-1] if self.drawedges else [])

    def _add_solids_patches(self, X, Y, C, mappable):
        hatches = mappable.hatches * len(C)  # Have enough hatches.
        patches = []
        for i in range(len(X) - 1):
            xy = np.array([[X[i, 0], Y[i, 0]],
                           [X[i, 1], Y[i, 0]],
                           [X[i + 1, 1], Y[i + 1, 0]],
                           [X[i + 1, 0], Y[i + 1, 1]]])
            patch = mpatches.PathPatch(mpath.Path(xy),
                                       facecolor=self.cmap(self.norm(C[i][0])),
                                       hatch=hatches[i], linewidth=0,
                                       antialiased=False, alpha=self.alpha)
            self.ax.add_patch(patch)
            patches.append(patch)
        self.solids_patches = patches

    def _do_extends(self, ax=None):
        """
        Add the extend tri/rectangles on the outside of the axes.

        ax is unused, but required due to the callbacks on xlim/ylim changed
        """
        # Clean up any previous extend patches
        for patch in self._extend_patches:
            patch.remove()
        self._extend_patches = []
        # extend lengths are fraction of the *inner* part of colorbar,
        # not the total colorbar:
        _, extendlen = self._proportional_y()
        bot = 0 - (extendlen[0] if self._extend_lower() else 0)
        top = 1 + (extendlen[1] if self._extend_upper() else 0)

        # xyout is the outline of the colorbar including the extend patches:
        if not self.extendrect:
            # triangle:
            xyout = np.array([[0, 0], [0.5, bot], [1, 0],
                              [1, 1], [0.5, top], [0, 1], [0, 0]])
        else:
            # rectangle:
            xyout = np.array([[0, 0], [0, bot], [1, bot], [1, 0],
                              [1, 1], [1, top], [0, top], [0, 1],
                              [0, 0]])
