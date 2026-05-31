            axis.update_units(data)

        mpl_scale.set_default_locators_and_formatters(axis)
        new._matplotlib_scale = mpl_scale

        normalize: Optional[Callable[[ArrayLike], ArrayLike]]
        if prop.normed:
            if new.norm is None:
                vmin, vmax = data.min(), data.max()
            else:
                vmin, vmax = new.norm
            vmin, vmax = axis.convert_units((vmin, vmax))
            a = forward(vmin)
            b = forward(vmax) - forward(vmin)

            def normalize(x):
                return (x - a) / b

        else:
            normalize = vmin = vmax = None

        new._pipeline = [
            axis.convert_units,
            forward,
            normalize,
            prop.get_mapping(new, data)
        ]

        def spacer(x):
            x = x.dropna().unique()
            if len(x) < 2:
                return np.nan
            return np.min(np.diff(np.sort(x)))
        new._spacer = spacer

        # TODO How to allow disabling of legend for all uses of property?
        # Could add a Scale parameter, or perhaps Scale.suppress()?
        # Are there other useful parameters that would be in Scale.legend()
        # besides allowing Scale.legend(False)?
        if prop.legend:
            axis.set_view_interval(vmin, vmax)
            locs = axis.major.locator()
            locs = locs[(vmin <= locs) & (locs <= vmax)]
            labels = axis.major.formatter.format_ticks(locs)
            new._legend = list(locs), list(labels)

        return new

    def _get_transform(self):

        arg = self.trans

        def get_param(method, default):
            if arg == method:
                return default
            return float(arg[len(method):])

        if arg is None:
            return _make_identity_transforms()
        elif isinstance(arg, tuple):
            return arg
        elif isinstance(arg, str):
            if arg == "ln":
                return _make_log_transforms()
            elif arg == "logit":
                base = get_param("logit", 10)
                return _make_logit_transforms(base)
            elif arg.startswith("log"):
                base = get_param("log", 10)
                return _make_log_transforms(base)
            elif arg.startswith("symlog"):
                c = get_param("symlog", 1)
                return _make_symlog_transforms(c)
            elif arg.startswith("pow"):
                exp = get_param("pow", 2)
                return _make_power_transforms(exp)
            elif arg == "sqrt":
                return _make_sqrt_transforms()
            else:
                raise ValueError(f"Unknown value provided for trans: {arg!r}")


@dataclass
class Continuous(ContinuousBase):
    """
    A numeric scale supporting norms and functional transforms.
