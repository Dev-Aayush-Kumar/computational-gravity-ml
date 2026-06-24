import sympy as sp

x = sp.symbols('x', positive=True)

phi_n = -1/(2*x)

phi_gr = sp.Rational(1,2) * (
    sp.sqrt(1 - 1/x) - 1
)

difference = phi_gr - phi_n

derivative = sp.simplify(
    sp.diff(difference, x)
)

critical_points = sp.solve(
    derivative,
    x
)

print("Critical Points:")
print(critical_points)