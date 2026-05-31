"""
Mathematica code printer
"""

from __future__ import print_function, division
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.str import StrPrinter
from sympy.printing.precedence import precedence

# Used in MCodePrinter._print_Function(self)
known_functions = {
    "exp": [(lambda x: True, "Exp")],
    "log": [(lambda x: True, "Log")],
    "sin": [(lambda x: True, "Sin")],
    "cos": [(lambda x: True, "Cos")],
    "tan": [(lambda x: True, "Tan")],
    "cot": [(lambda x: True, "Cot")],
    "asin": [(lambda x: True, "ArcSin")],
    "acos": [(lambda x: True, "ArcCos")],
    "atan": [(lambda x: True, "ArcTan")],
    "sinh": [(lambda x: True, "Sinh")],
    "cosh": [(lambda x: True, "Cosh")],
    "tanh": [(lambda x: True, "Tanh")],
    "coth": [(lambda x: True, "Coth")],
    "sech": [(lambda x: True, "Sech")],
    "csch": [(lambda x: True, "Csch")],
    "asinh": [(lambda x: True, "ArcSinh")],
    "acosh": [(lambda x: True, "ArcCosh")],
    "atanh": [(lambda x: True, "ArcTanh")],
    "acoth": [(lambda x: True, "ArcCoth")],
    "asech": [(lambda x: True, "ArcSech")],
    "acsch": [(lambda x: True, "ArcCsch")],
    "conjugate": [(lambda x: True, "Conjugate")],

}


class MCodePrinter(CodePrinter):
    """A printer to convert python expressions to
    strings of the Wolfram's Mathematica code
    """
    printmethod = "_mcode"

    _default_settings = {
        'order': None,
        'full_prec': 'auto',
        'precision': 15,
        'user_functions': {},
        'human': True,
        'allow_unknown_functions': False,
    }

    _number_symbols = set()
    _not_supported = set()

    def __init__(self, settings={}):
        """Register function mappings supplied by user"""
        CodePrinter.__init__(self, settings)
        self.known_functions = dict(known_functions)
        userfuncs = settings.get('user_functions', {})
        for k, v in userfuncs.items():
            if not isinstance(v, list):
                userfuncs[k] = [(lambda *x: True, v)]
                self.known_functions.update(userfuncs)

    doprint = StrPrinter.doprint

    def _print_Pow(self, expr):
        PREC = precedence(expr)
        return '%s^%s' % (self.parenthesize(expr.base, PREC),
                          self.parenthesize(expr.exp, PREC))

    def _print_Mul(self, expr):
        PREC = precedence(expr)
        c, nc = expr.args_cnc()
        res = super(MCodePrinter, self)._print_Mul(expr.func(*c))
        if nc:
