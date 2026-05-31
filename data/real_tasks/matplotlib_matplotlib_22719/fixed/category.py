import numpy as np

from matplotlib import _api, ticker, units


_log = logging.getLogger(__name__)


class StrCategoryConverter(units.ConversionInterface):
    @staticmethod
    def convert(value, unit, axis):
        """
        Convert strings in *value* to floats using mapping information stored
        in the *unit* object.

        Parameters
        ----------
        value : str or iterable
            Value or list of values to be converted.
        unit : `.UnitData`
            An object mapping strings to integers.
        axis : `~matplotlib.axis.Axis`
            The axis on which the converted value is plotted.

            .. note:: *axis* is unused.

        Returns
        -------
        float or ndarray[float]
        """
        if unit is None:
            raise ValueError(
                'Missing category information for StrCategoryConverter; '
                'this might be caused by unintendedly mixing categorical and '
                'numeric data')
        StrCategoryConverter._validate_unit(unit)
        # dtype = object preserves numerical pass throughs
        values = np.atleast_1d(np.array(value, dtype=object))
        # pass through sequence of non binary numbers
        with _api.suppress_matplotlib_deprecation_warning():
            is_numlike = all(units.ConversionInterface.is_numlike(v)
                             and not isinstance(v, (str, bytes))
                             for v in values)
        if values.size and is_numlike:
            _api.warn_deprecated(
                "3.5", message="Support for passing numbers through unit "
                "converters is deprecated since %(since)s and support will be "
                "removed %(removal)s; use Axis.convert_units instead.")
            return np.asarray(values, dtype=float)
        # force an update so it also does type checking
        unit.update(values)
        return np.vectorize(unit._mapping.__getitem__, otypes=[float])(values)

    @staticmethod
    def axisinfo(unit, axis):
        """
        Set the default axis ticks and labels.

        Parameters
        ----------
        unit : `.UnitData`
            object string unit information for value
        axis : `~matplotlib.axis.Axis`
            axis for which information is being set

            .. note:: *axis* is not used

        Returns
        -------
        `~matplotlib.units.AxisInfo`
            Information to support default tick labeling

        """
        StrCategoryConverter._validate_unit(unit)
        # locator and formatter take mapping dict because
        # args need to be pass by reference for updates
        majloc = StrCategoryLocator(unit._mapping)
        majfmt = StrCategoryFormatter(unit._mapping)
        return units.AxisInfo(majloc=majloc, majfmt=majfmt)

    @staticmethod
    def default_units(data, axis):
        """
        Set and update the `~matplotlib.axis.Axis` units.

        Parameters
        ----------
