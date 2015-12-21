import math
from collections import Counter

def build_vector(iterable1, iterable2):
    counter1 = Counter(iterable1)
    counter2 = Counter(iterable2)
    all_items = set(counter1.keys()).union(set(counter2.keys()))
    vector1 = [counter1[k] for k in all_items]
    vector2 = [counter2[k] for k in all_items]
    return vector1, vector2

def cosim(v1, v2):
    dot_product = sum(n1 * n2 for n1, n2 in zip(v1, v2) )
    magnitude1 = math.sqrt(sum(n ** 2 for n in v1))
    magnitude2 = math.sqrt(sum(n ** 2 for n in v2))
    return dot_product / (magnitude1 * magnitude2)

v_a = [1,0,1,0,0,0,0]
v_b = [0,1,0,0,0,0,0]
v_c = [1,1,0,0,0,0,0]
v_d = [1,1,0,1,1,1,1]
v_e = [4,1,0,1,1,1,1]
print(v_a)
print(v_b)
print(v_c)
print(v_d)
print(v_e)

print(cosim(v_a,v_b)
print(cosim(v_a,v_c)
print(cosim(v_a,v_d)
print(cosim(v_a,v_e)
v1 = [1,1,1,1,1,1]
v2 = [4,0,0,0,0,0]
v3 = [1,1,1,1,0,0]
v4 = [1,0,0,0,0,0]

print(v1)
print(v2)
print(v3)
print(v4)


print(cosim(v1,v2)
print(cosim(v1,v3)
print(cosim(v1,v4)

