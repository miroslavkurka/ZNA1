import unittest
import numpy as np
import object_intersection as ob
import rice_siff as rs


class TestObjectIntersection(unittest.TestCase):

    def test_object_intersection_size_0(self):
        n = 0
        matrix = np.ones((n, n))
        np.fill_diagonal(matrix, 0)
        result = ob.object_intersection(matrix)
        expected_length = 2 ** n
        self.assertEqual(len(result), expected_length)

    def test_object_intersection_size_1(self):
        n = 1
        matrix = np.ones((n, n))
        np.fill_diagonal(matrix, 0)
        result = ob.object_intersection(matrix)
        expected_length = 2 ** n
        self.assertEqual(len(result), expected_length)

    def test_object_intersection_size_2(self):
        n = 2
        matrix = np.ones((n, n))
        np.fill_diagonal(matrix, 0)
        result = ob.object_intersection(matrix)
        expected_length = 2 ** n
        self.assertEqual(len(result), expected_length)

    def test_object_intersection_large_size(self):
        n = 5
        matrix = np.ones((n, n))
        np.fill_diagonal(matrix, 0)
        result = ob.object_intersection(matrix)
        expected_length = 2 ** n
        self.assertEqual(len(result), expected_length)

    def test_object_intersection_non_square_input(self):
        with self.assertRaises(ValueError):
            matrix = np.ones((3, 4))  # Non-square matrix
            np.fill_diagonal(matrix, 0)
            ob.object_intersection(matrix)

    def test_object_intersection_sparse_matrix(self):
        n = 5
        matrix = np.zeros((n, n))
        np.fill_diagonal(matrix, 0)
        result = ob.object_intersection(matrix)
        expected_length = 2 ** n
        self.assertNotEqual(len(result), expected_length)

    def test_object_intersection_dense_matrix(self):
        n = 5
        matrix = np.ones((n, n))
        result = ob.object_intersection(matrix)
        expected_length = 2 ** n
        self.assertNotEqual(len(result), expected_length)

    def test_object_intersection_degenerate_case(self):
        n = 4
        matrix = np.ones((n, n))
        matrix[1, 1] = matrix[2, 2] = 0
        result = ob.object_intersection(matrix)
        expected_length = 2 ** n
        self.assertNotEqual(len(result), expected_length)

    def test_object_intersection_boundary_condition(self):
        n = 16
        matrix = np.ones((n, n))
        np.fill_diagonal(matrix, 0)
        result = ob.object_intersection(matrix)
        expected_length = 2 ** n
        self.assertEqual(len(result), expected_length)


class TestRiceSiff(unittest.TestCase):

    def test_rice_siff_size_0(self):
        n = 0
        matrix = np.ones((n, n))
        np.fill_diagonal(matrix, 0)
        result = rs.rice_siff(matrix)
        expected_length = n
        self.assertEqual(len(result), expected_length)

    def test_rice_siff_size_1(self):
        n = 1
        matrix = np.ones((n, n))
        np.fill_diagonal(matrix, 0)
        result = rs.rice_siff(matrix)
        expected_length = n
        self.assertEqual(len(result), expected_length)

    def test_rice_siff_size_2(self):
        n = 2
        matrix = np.ones((n, n))
        np.fill_diagonal(matrix, 0)
        result = rs.rice_siff(matrix)
        expected_length = n
        self.assertEqual(len(result), expected_length)

    def test_rice_siff_large_size(self):
        n = 5
        matrix = np.ones((n, n))
        np.fill_diagonal(matrix, 0)
        result = rs.rice_siff(matrix)
        expected_length = n
        self.assertEqual(len(result), expected_length)

    def test_rice_siff_non_square_input(self):
        with self.assertRaises(ValueError):
            matrix = np.ones((3, 4))  # Non-square matrix
            np.fill_diagonal(matrix, 0)
            rs.rice_siff(matrix)

    def test_rice_siff_sparse_matrix(self):
        n = 5
        matrix = np.zeros((n, n))
        np.fill_diagonal(matrix, 0)
        result = rs.rice_siff(matrix)
        expected_length = n
        self.assertNotEqual(len(result), expected_length)


if __name__ == '__main__':
    unittest.main()
