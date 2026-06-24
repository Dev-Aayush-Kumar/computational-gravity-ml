import sympy as sp

x = sp.symbols('x', positive=True)

phi_n = -1/(2*x)

phi_gr = sp.Rational(1,2) * (
    sp.sqrt(1 - 1/x) - 1
)

difference = phi_gr - phi_n

error = 100 * (difference / phi_gr)

error = sp.simplify(error)
error = sp.factor(error)

print("\n===== SIMPLIFIED ERROR =====\n")
print(error)

print("\n===== PRETTY FORM =====\n")
sp.pprint(error)