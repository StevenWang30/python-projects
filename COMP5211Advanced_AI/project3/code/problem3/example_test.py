import IPython
import numpy as np

with open("./Data/samples.txt", encoding="utf8") as f:
    lines = [line.strip() for line in f]
qa_pairs = [line.split('|||') for line in lines]

# IPython.embed()

from bert_serving.client import BertClient
bc = BertClient() # on your computer 
# bc = BertClient(ip="lgpu1", port=10086, port_out=10087) # on CSD computer

keys = [qa[0] for qa in qa_pairs]
print("The question list: ")
for i in range(len(keys)):
    print(keys[i])
    print("\n")
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
    return qa_pairs[min_idx]

save_file = "./result1.txt"
f = open(save_file, "w")
f.close()
while True:
    query = input(">>> (input your query):")
    min_qa_pair = search_vec([query])

    print("If you mean this?: ", min_qa_pair[0].strip())
    print("The answer is:\n" + min_qa_pair[1].strip())
    f = open(save_file, "a+")
    f.write("Query: \n")
    f.write("  " + query.strip())
    f.write("\n")
    f.write("Possible problem: \n")
    f.write("  " + min_qa_pair[0].strip())
    f.write("\n")
    f.write("Answer: \n")
    f.write("  " + min_qa_pair[1].strip())
    f.write("\n")
    f.write("\n")
    f.close()


    
