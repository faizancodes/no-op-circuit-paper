        offsets = (maxh + sep) * np.arange(len(widths))
        return total, offsets


def _get_aligned_offsets(hd_list, height, align="baseline"):
    """
    Align boxes each specified by their ``(height, descent)`` pair.

    For simplicity of the description, the terminology used here assumes a
    horizontal layout (i.e., vertical alignment), but the function works
    equally for a vertical layout.

    Parameters
    ----------
    hd_list
        List of (height, xdescent) of boxes to be aligned.
    height : float or None
        Intended total height. If None, the maximum of the heights in *hd_list*
        is used.
    align : {'baseline', 'left', 'top', 'right', 'bottom', 'center'}
        The alignment anchor of the boxes.

    Returns
    -------
    height
        The total height of the packing (if a value was originally passed in,
        it is returned without checking that it is actually large enough).
    descent
        The descent of the packing.
    offsets
        The bottom offsets of the boxes.
    """

    if height is None:
        height = max(h for h, d in hd_list)
    _api.check_in_list(
        ["baseline", "left", "top", "right", "bottom", "center"], align=align)

    if align == "baseline":
        height_descent = max(h - d for h, d in hd_list)
        descent = max(d for h, d in hd_list)
        height = height_descent + descent
        offsets = [0. for h, d in hd_list]
    elif align in ["left", "bottom"]:
        descent = 0.
        offsets = [d for h, d in hd_list]
    elif align in ["right", "top"]:
        descent = 0.
        offsets = [height - h + d for h, d in hd_list]
    elif align == "center":
        descent = 0.
        offsets = [(height - h) * .5 + d for h, d in hd_list]

    return height, descent, offsets


class OffsetBox(martist.Artist):
    """
    The OffsetBox is a simple container artist.

    The child artists are meant to be drawn at a relative position to its
    parent.

    Being an artist itself, all parameters are passed on to `.Artist`.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self._internal_update(kwargs)
        # Clipping has not been implemented in the OffsetBox family, so
        # disable the clip flag for consistency. It can always be turned back
        # on to zero effect.
        self.set_clip_on(False)
        self._children = []
        self._offset = (0, 0)

    def set_figure(self, fig):
        """
        Set the `.Figure` for the `.OffsetBox` and all its children.

        Parameters
        ----------
        fig : `~matplotlib.figure.Figure`
        """
        super().set_figure(fig)
        for c in self.get_children():
            c.set_figure(fig)

    @martist.Artist.axes.setter
    def axes(self, ax):
        # TODO deal with this better
