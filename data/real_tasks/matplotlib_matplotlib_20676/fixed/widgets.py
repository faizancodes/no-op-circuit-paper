
    rect = _api.deprecate_privatize_attribute("3.5")

    rectprops = _api.deprecate_privatize_attribute("3.5")

    active_handle = _api.deprecate_privatize_attribute("3.5")

    pressv = _api.deprecate_privatize_attribute("3.5")

    span_stays = _api.deprecated("3.5")(
        property(lambda self: self._interactive)
        )

    prev = _api.deprecate_privatize_attribute("3.5")

    def new_axes(self, ax):
        """Set SpanSelector to operate on a new Axes."""
        self.ax = ax
        if self.canvas is not ax.figure.canvas:
            if self.canvas is not None:
                self.disconnect_events()

            self.canvas = ax.figure.canvas
            self.connect_default_events()

        if self.direction == 'horizontal':
            trans = ax.get_xaxis_transform()
            w, h = 0, 1
        else:
            trans = ax.get_yaxis_transform()
            w, h = 1, 0
        self._rect = Rectangle((0, 0), w, h,
                               transform=trans,
                               visible=False,
                               **self._rectprops)

        self.ax.add_patch(self._rect)
        if len(self.artists) > 0:
            self.artists[0] = self._rect
        else:
            self.artists.append(self._rect)

    def _setup_edge_handle(self, props):
        # Define initial position using the axis bounds to keep the same bounds
        if self.direction == 'horizontal':
            positions = self.ax.get_xbound()
        else:
            positions = self.ax.get_ybound()
        self._edge_handles = ToolLineHandles(self.ax, positions,
                                             direction=self.direction,
                                             line_props=props,
                                             useblit=self.useblit)
        self.artists.extend([line for line in self._edge_handles.artists])

    def _press(self, event):
        """Button press event handler."""
        if self._interactive and self._rect.get_visible():
            self._set_active_handle(event)
        else:
            self._active_handle = None

        if self._active_handle is None or not self._interactive:
            # Clear previous rectangle before drawing new rectangle.
            self.update()

        v = event.xdata if self.direction == 'horizontal' else event.ydata
        # self._pressv and self._prev are deprecated but we still need to
        # maintain them
        self._pressv = v
        self._prev = self._get_data(event)

        if self._active_handle is None:
            # when the press event outside the span, we initially set the
            # visibility to False and extents to (v, v)
            # update will be called when setting the extents
            self.visible = False
            self.extents = v, v
            # We need to set the visibility back, so the span selector will be
            # drawn when necessary (span width > 0)
            self.visible = True
        else:
            self.set_visible(True)

        return False

    @property
    def direction(self):
        """Direction of the span selector: 'vertical' or 'horizontal'."""
        return self._direction

    @direction.setter
    def direction(self, direction):
