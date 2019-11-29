from z3 import *
import IPython

L = Int('Lisa')
B = Int('Bob')
J = Int('Jim')
M = Int('Mary')

s = Solver()

s.add(L > 0, L <= 4, B > 0, B <= 4, J > 0, J <= 4, M > 0, M <= 4)
s.add(L != B, L != J, L != M, B != J, B != M, J != M)
s.add(L - B != 1, B - L != 1)
s.add(Or(J < L, J < M))
s.add(J - B == 1)
s.add(Or(L == 1, M == 1))

s.check()
s.model()

print("Now check the solver: \n", s.check())
print("The solution is: \n", s.model())

# IPython.embed()
# # the following is a python
# # code to compute a model of p ∨ q and ¬(p ∧ q):
# p = Bool('p')
# q = Bool('q')
# s = Solver()
# s.add(Or(p,q))
# s.add(Not(And(p,q)))
# print(s.check())
# print(s.model())
