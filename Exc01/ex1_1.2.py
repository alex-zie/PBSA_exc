import numpy as np

arr1 = np.array([[8,5,6],[5,6,4],[6,4,7]])
arr2 = np.array([[6,-3,2],[3,5,1],[-2,-1,4]])
arr3 = np.array([[3,5,6],[21,35,42],[27,45,54]])
arr4 = np.array([[7,4,-2],[4,8,3],[-2,3,6]])
arr5 = np.diag([8,-1,7])
arr6 = np.array([[6,3,-3],[3,6,3],[-3,3,6]])
arr7 = np.array([[-3,1,1],[1,-3,1],[1,1,-3]])

# check symmetry
def checkSymm(arr):
    # return (arr == arr.T).all()
    arr_transp = arr.T
    for i in range(len(arr)):
        for j in range(len(arr_transp):
            if not arr[i,j] == arr_transp[i,j]:
                return False
    return True

arr1symm = checkSymm(arr1)
arr2symm = checkSymm(arr2)
arr3symm = checkSymm(arr3)
arr4symm = checkSymm(arr4)
arr5symm = checkSymm(arr5)
arr6symm = checkSymm(arr6)
arr7symm = checkSymm(arr7)

# check diagonal dominance
def checkDiagDom(arr):
    # abs_arr = np.abs(arr)
    # return np.all(2 * np.diag(abs_arr) >= np.sum(abs_arr, axis=1))
    main_elem = 0
    elem_sum = 0
    for i in range(len(arr)):
        main_elem = np.abs(arr[i,i])
        for j in range(len(arr)):
            elem_sum += np.abs(arr[i,j])
        if elem_sum > main_elem:
            return 'Not diagonally dominant'
    if main_elem > elem_sum:
        return 'Strictly diagonally dominant'
    return 'Diagonally dominant'
              

arr1diagdom = checkDiagDom(arr1)
arr2diagdom = checkDiagDom(arr2)
arr3diagdom = checkDiagDom(arr3)
arr4diagdom = checkDiagDom(arr4)
arr5diagdom = checkDiagDom(arr5)
arr6diagdom = checkDiagDom(arr6)
arr7diagdom = checkDiagDom(arr7)

# check definiteness
def checkDef(arr):
    if checkSymm(arr) is False:
        return 'Indefinite'
    evals, evecs = np.linalg.eig(arr) # probably illegal
    if (evals > 0).all():
        return 'Symmetrically positive definite'
    if (evals >= 0).all():
        return 'Symmetrically positive semi-definite'
    if (evals < 0).all():
        return 'Symmetrically negative definite'
    if (evals <= 0).all():
        return 'Symmetrically negative semi-definite'
    return 'Error'

print(checkDef(arr1))
