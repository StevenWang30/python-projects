from z3 import *
solver = Solver()
a = Int('a')
b = Int('b')
solver.add(a*b==0x24)
if solver.check()==sat:
	m = solver.model()
	print(m)