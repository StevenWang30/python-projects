from z3 import *
import IPython


# L1 means lady in room 1 and Not(L1) means tiger in room 1
L1 = Bool('L1')
L2 = Bool('L2')
L3 = Bool('L3')
R1 = Not(L1)
R2 = L2
R3 = Not(L2)

s = Solver()

# one room contained a lady and the other two contained tigers
# s.add(Implies(L1, Not(Or(L2, L3))))
# s.add(Implies(L2, Not(Or(L1, L3))))
# s.add(Implies(L3, Not(Or(L1, L2))))
# s.add(Or(L1, L2, L3))
s.add(AtMost(L1, L2, L3, 1))
s.add(AtLeast(L1, L2, L3, 1))

# three signs
s.add(AtMost(R1, R2, R3, 1))
# s.add(Implies(R1, Not(Or(R2, R3))))
# s.add(Implies(R2, Not(Or(R1, R3))))
# s.add(Implies(R3, Not(Or(R1, R2))))

# check three situations
s1 = s.translate(main_ctx())
s1.add(L1)
s2 = s.translate(main_ctx())
s2.add(L2)
s3 = s.translate(main_ctx())
s3.add(L3)

if s1.check() == sat:
	print("Lady can stay in Room 1.")
else:
	print("Lady can not stay in Room 1")

if s2.check() == sat:
	print("Lady can stay in Room 2.")
else:
	print("Lady can not stay in Room 2")

if s3.check() == sat:
	print("Lady can stay in Room 3.")
else:
	print("Lady can not stay in Room 3")


# for Problem 1: lady.py, which is a z3 query file for your answer. For
# example, if your answer is that the lady is in room 2, and you use l2 to
# denote it, then your z3 query file is to check whether not(l2) is consistent
# with the KB of this problem. For example, the following python code is a
# z3 query file to check if q follows from p and p ⊃ q:

# # L1 means lady in room 1 and Not(L1) means tiger in room 1
# L1 = Bool('L1')
# L2 = Bool('L2')
# L3 = Bool('L3')
# R1 = Bool('R1')
# R2 = Bool('R2')
# R3 = Bool('R3')

# s = Solver()

# # one room contained a lady and the other two contained tigers
# s.add(Implies(L1, Not(Or(L2, L3))))
# s.add(Implies(L2, Not(Or(L1, L3))))
# s.add(Implies(L3, Not(Or(L1, L2))))
# s.add(Or(L1, L2, L3))

# # three signs
# # s.add(Implies(R1, Not(L1)))
# # s.add(Implies(R2, L2))
# # s.add(Implies(R3, Not(L3)))
# # s.add(AtMost(R1, R2, R3, 1))

# s.add(Implies(L1, Not(R1)))
# s.add(Implies(L2, R2))
# s.add(Implies(L2, Not(R3)))
# s.add(AtMost(R1, R2, R3, 1))

# # check three situations
# if s.check(L1,Not(L2),Not(L3)):
# 	print("Lady in Room 1.")

# if s.check(L2,Not(L1),Not(L3)):
# 	print("Lady in Room 2.")

# if s.check(L3,Not(L2),Not(L1)):
# 	print("Lady in Room 3.")

# IPython.embed()



