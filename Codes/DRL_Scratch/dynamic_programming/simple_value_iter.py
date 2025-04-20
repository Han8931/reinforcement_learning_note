"""
gamma: 0.9
states: L1 and L2 
actions: Left or Right
Rewards:
    - L1->L->L1->-1
    - L1->R->L2->1
    - L2->L->L1->0
    - L2->R->L2->-1
"""

V = {'L1': 0.0, 'L2':0.0}
new_V = V.copy()

for _ in range(100):
    new_V['L1'] = 0.5*(-1+0.9*V['L1'])+0.5*(1+0.9*V['L2'])
    new_V['L2'] = 0.5*(0+0.9*V['L1'])+0.5*(-1+0.9*V['L2'])
    V = new_V.copy()
    # print(V)

V = {'L1': 0.0, 'L2':0.0}
new_V = V.copy()

cnt = 0
while True:
    new_V['L1'] = 0.5*(-1+0.9*V['L1'])+0.5*(1+0.9*V['L2'])
    new_V['L2'] = 0.5*(0+0.9*V['L1'])+0.5*(-1+0.9*V['L2'])
    delta = abs(new_V['L1']-V['L1'])
    delta = max(delta, abs(new_V['L2']-V['L2']))
    V = new_V.copy()
    cnt+=1
    if delta<0.0001:
        print(V)
        print(f"Count: {cnt}")
        break

V = {'L1': 0.0, 'L2':0.0}

cnt = 0
while True:
    t = 0.5*(-1+0.9*V['L1'])+0.5*(1+0.9*V['L2'])
    delta = abs(t-V['L1'])
    V['L1'] = t # V1 is updated

    t = 0.5*(0+0.9*V['L1'])+0.5*(-1+0.9*V['L2'])
    delta = max(delta, abs(t-V['L2']))
    V['L2'] = t
    cnt+=1
    if delta<0.0001:
        print(V)
        print(f"Count: {cnt}")
        break
