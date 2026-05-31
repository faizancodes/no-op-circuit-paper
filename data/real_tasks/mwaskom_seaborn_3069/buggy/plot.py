
        base_legend = None
        for (name, _), (handles, labels) in merged_contents.items():

            legend = mpl.legend.Legend(
                self._figure,
                handles,
                labels,
                title=name,
                loc=loc,
                bbox_to_anchor=(.98, .55),
            )

            if base_legend:
                # Matplotlib has no public API for this so it is a bit of a hack.
                # Ideally we'd define our own legend class with more flexibility,
                # but that is a lot of work!
                base_legend_box = base_legend.get_children()[0]
                this_legend_box = legend.get_children()[0]
                base_legend_box.get_children().extend(this_legend_box.get_children())
            else:
                base_legend = legend
                self._figure.legends.append(legend)

    def _finalize_figure(self, p: Plot) -> None:

        for sub in self._subplots:
            ax = sub["ax"]
            for axis in "xy":
                axis_key = sub[axis]

                # Axis limits
                if axis_key in p._limits:
                    convert_units = getattr(ax, f"{axis}axis").convert_units
                    a, b = p._limits[axis_key]
                    lo = a if a is None else convert_units(a)
                    hi = b if b is None else convert_units(b)
                    if isinstance(a, str):
                        lo = cast(float, lo) - 0.5
                    if isinstance(b, str):
                        hi = cast(float, hi) + 0.5
                    ax.set(**{f"{axis}lim": (lo, hi)})

        engine_default = None if p._target is not None else "tight"
        layout_engine = p._layout_spec.get("engine", engine_default)
        set_layout_engine(self._figure, layout_engine)
