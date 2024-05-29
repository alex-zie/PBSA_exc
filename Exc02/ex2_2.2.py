import numpy as np

# 2.2a)
def rotation_matrix(angle, axis):
    sin = np.sin(angle)
    cos = np.cos(angle)
    if axis == 'x':
        R = np.array([[1, 0, 0, 0], [0, cos, -sin, 0], [0, sin, cos, 0], [0, 0, 0, 1]])
    elif axis == 'y':
        R = np.array([[cos, 0, sin, 0], [0, 1, 0, 0], [-sin, 0, cos, 0], [0, 0, 0, 1]])
    else:
        R = np.array([[cos, -sin, 0, 0], [sin, cos, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    return R

def translation_matrix(x, y, z):
    return np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])

mat_1 = rotation_matrix(np.pi/2, 'z')
mat_2 = translation_matrix(0, -2, 1) @ rotation_matrix(np.pi/2, 'x')
mat_3 = rotation_matrix(np.pi/2, 'z') @ translation_matrix(-1, -1, -2)
mat_4 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 1/1, 1]])
# Transformationsmatrizen
print(mat_1)
print(mat_2)
print(mat_3)
print(mat_4)

# Test mit einem Vektor; Ergebnisse sinnvoll
test_vec = np.array([[0], [1], [0], [1]])
print(mat_1 @ test_vec) # expected: (-1, 0, 0)
print(mat_2 @ test_vec) # expected: (0, -2, 2)
print(mat_3 @ test_vec) # expected: (0, -1, -2)
print(mat_4 @ test_vec) # expected: 

# 2.2b)
v = np.reshape([1, 0, 3], (3, 1))
r = np.array([0, 1, 0])

def rodrigues_formel(v, r, theta):
    return v * np.cos(theta) + (1 - np.cos(theta)) * (r @ v) * np.reshape(r, (3, 1)) + np.reshape(np.cross(r, np.reshape(v, (1, 3))), (3, 1)) * np.sin(theta)

print(rodrigues_formel(v, r, np.pi/2))
