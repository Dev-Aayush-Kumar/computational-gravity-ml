import sympy as sp

x = sp.symbols('x', positive=True)

phi_n = -1/(2*x)

phi_gr = sp.Rational(1,2) * (
    sp.sqrt(1 - 1/x) - 1
)

ratio = sp.simplify(
    sp.limit(
        phi_gr/phi_n,
        x,
        sp.oo
    )
)

print(
    "\nWeak-field ratio:"
)

print(ratio)