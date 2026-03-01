import numpy as np

def get_adjoint(T):
    R = T[0:3, 0:3]
    p = T[0:3, 3]

    p_skew = np.array([
        [0, -p[2], p[1]],
        [p[2], 0, -p[0]],
        [-p[1], p[0], 0]
    ])

    adj = np.zeros((6, 6))
    adj[0:3, 0:3] = R
    adj[3:6, 3:6] = R
    adj[3:6, 0:3] = np.dot(p_skew, R)
    
    return adj

T_sb = np.array([[1, 0, 0, 10],[0, 0.9848, -0.1736, 5],[0, 0.1736, 0.9848, 0],[0, 0, 0, 1]])
T_bc = np.array([[1, 0, 0, 1], [0, 1, 0, 0],[0, 0, 1, 2],[0, 0, 0, 1]])
T_sc = np.dot(T_sb, T_bc)

#  드론 위치 추적 (P_s=T_sc*P_c)
p_drone_c = np.array([0, 0, 5, 1])
p_drone_s = np.dot(T_sc, p_drone_c)

# 드론 속도 추적 
V_c = np.array([0, 0, 0, 0, 0, 1]) 
Ad_Tsc = get_adjoint(T_sc)
V_s = np.dot(Ad_Tsc, V_c)

print(f"드론의 지구 기준 위치: {p_drone_s[:3]}")
print(f"드론의 지구 기준 속도: {V_s[3:]}")
