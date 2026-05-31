"""
Python code printers

This module contains Python code printers for plain Python as well as NumPy & SciPy enabled code.
"""
from collections import defaultdict
from itertools import chain
from sympy.core import S
from .precedence import precedence
from .codeprinter import CodePrinter

_kw = {
    'and', 'as', 'assert', 'break', 'class', 'continue', 'def', 'del', 'elif',
    'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in',
    'is', 'lambda', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while',
    'with', 'yield', 'None', 'False', 'nonlocal', 'True'
}

_known_functions = {
    'Abs': 'abs',
}
_known_functions_math = {
    'acos': 'acos',
    'acosh': 'acosh',
    'asin': 'asin',
    'asinh': 'asinh',
    'atan': 'atan',
    'atan2': 'atan2',
    'atanh': 'atanh',
    'ceiling': 'ceil',
    'cos': 'cos',
    'cosh': 'cosh',
    'erf': 'erf',
    'erfc': 'erfc',
    'exp': 'exp',
    'expm1': 'expm1',
    'factorial': 'factorial',
    'floor': 'floor',
    'gamma': 'gamma',
    'hypot': 'hypot',
    'loggamma': 'lgamma',
    'log': 'log',
    'ln': 'log',
    'log10': 'log10',
    'log1p': 'log1p',
    'log2': 'log2',
    'sin': 'sin',
    'sinh': 'sinh',
    'Sqrt': 'sqrt',
    'tan': 'tan',
    'tanh': 'tanh'
}  # Not used from ``math``: [copysign isclose isfinite isinf isnan ldexp frexp pow modf
# radians trunc fmod fsum gcd degrees fabs]
_known_constants_math = {
    'Exp1': 'e',
    'Pi': 'pi',
    'E': 'e',
    'Infinity': 'inf',
    'NaN': 'nan',
    'ComplexInfinity': 'nan'
}

def _print_known_func(self, expr):
