import sympy as sp

x = sp.symbols('x', positive=True)

error = 100*(
    (-x**sp.Rational(3,2)
     + sp.sqrt(x)
     + x*sp.sqrt(x-1))
    /
    (x*(-sp.sqrt(x)+sp.sqrt(x-1)))
)

print(
    sp.simplify(
        sp.diff(error,x)
    )
)