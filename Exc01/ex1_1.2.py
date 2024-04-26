import numpy as np

arr1 = np.array([[8,5,6],[5,6,4],[6,4,7]])
arr2 = np.array([[6,-3,2],[3,5,1],[-2,-1,4]])
arr3 = np.array([[3,5,6],[21,35,42],[27,45,54]])
arr4 = np.array([[7,4,-2],[4,8,3],[-2,3,6]])
arr5 = np.diag([8,-1,7])
arr6 = np.array([[6,3,-3],[3,6,3],[-3,3,6]])
arr7 = np.array([[-3,1,1],[1,-3,1],[1,1,-3]])

# check symmetry
def check_symm(arr):
    # return (arr == arr.T).all()
    arr_transp = arr.T
    for i in range(len(arr)):
        for j in range(len(arr_transp)):
            if arr[i,j] != arr_transp[i,j]:
                return False
    return True

arr1symm = check_symm(arr1)
arr2symm = check_symm(arr2)
arr3symm = check_symm(arr3)
arr4symm = check_symm(arr4)
arr5symm = check_symm(arr5)
arr6symm = check_symm(arr6)
arr7symm = check_symm(arr7)

# check diagonal dominance
def check_diag_dom(arr):
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
              

arr1diagdom = check_diag_dom(arr1)
arr2diagdom = check_diag_dom(arr2)
arr3diagdom = check_diag_dom(arr3)
arr4diagdom = check_diag_dom(arr4)
arr5diagdom = check_diag_dom(arr5)
arr6diagdom = check_diag_dom(arr6)
arr7diagdom = check_diag_dom(arr7)

# check definiteness
def check_def(arr):
    if check_symm(arr) is False:
        return 'Indefinite'
    evals, _ = np.linalg.eig(arr) # probably illegal
    if (evals > 0).all():
        return 'Symmetrically positive definite'
    if (evals >= 0).all():
        return 'Symmetrically positive semi-definite'
    if (evals < 0).all():
        return 'Symmetrically negative definite'
    if (evals <= 0).all():
        return 'Symmetrically negative semi-definite'
    return 'Error'
    # can be done w/o libraries but big headache therefore i'm leaving it as is
    # msg = ''
    # # Hauptminoren
    # hauptminoren = np.zeros([len(arr), 1])
    # for i in range(len(arr)):
    #     hauptminoren[i] = np.linalg.det(arr[:i+1, :i+1])
    # print(hauptminoren)
    # if (hauptminoren > 0).all():
    #     return 'Symmetrically positive definite'
    # for j in range(len(hauptminoren)):
    #     if (j % 2 == 0 and hauptminoren[j] <= 0) or (j % 2 == 1 and hauptminoren[j] >= 0):
    #         msg = ''
        

print(check_def(arr1))
