import IPython
import numpy as np

with open("./Data/NOT_test.txt", encoding="utf8") as f:
    lines = [line.strip() for line in f]

from bert_serving.client import BertClient
bc = BertClient() # on your computer 

keys = lines
print("The sentence list: ")
for i in range(len(keys)):
	print(keys[i])
key_vecs = bc.encode(keys)[:,0,:] # take [CLS] to represent the whole sentence```

def search_vec(query):
    def distance(a, b):
        return np.linalg.norm(a-b)
    q_v = bc.encode(query)[:,0,:]
    min_idx = -1
    min_dis = 1e100
    for i, k_v in enumerate(key_vecs):
        dist = distance(q_v, k_v)
        if dist < min_dis:
            min_dis = dist
            min_idx = i
    return lines[min_idx]

while True:
    query = input(">>> (input your query):")
    best_choice = search_vec([query])

    print("The cloest meaning sentence is:\n" + best_choice)

    
