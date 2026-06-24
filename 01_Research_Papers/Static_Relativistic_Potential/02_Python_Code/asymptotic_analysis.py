import sympy as sp

x = sp.symbols('x', positive=True)

phi_n = -1/(2*x)

phi_gr = sp.Rational(1,2) * (
    sp.sqrt(1 - 1/x) - 1
)

difference = sp.simplify(
    phi_gr - phi_n
)

series_expansion = sp.series(
    difference,
    x,
    sp.oo,
    5
)

print("\n===== ASYMPTOTIC EXPANSION =====\n")

print(series_expansion)