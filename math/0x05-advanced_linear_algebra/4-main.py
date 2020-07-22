#!/usr/bin/env python3

if __name__ == '__main__':
    inverse = __import__('4-inverse').inverse

    mat1 = [[5]]
    mat2 = [[1, 2], [3, 4]]
    mat3 = [[1, 1], [1, 1]]
    mat4 = [[5, 7, 9], [3, 1, 8], [6, 2, 4]]
    mat5 = []
    mat6 = [[1, 2, 3], [4, 5, 6]]

    print(inverse(mat1))
    print(inverse(mat2))
    print(inverse(mat3))
    print(inverse(mat4))
    try:
        inverse(mat5)
    except Exception as e:
        print(e)
    try:
        inverse(mat6)
    except Exception as e:
        print(e)
