import numpy as np

# Np arrays
list = [1, 2, 3, 4, 5]
print(type(list))
print(list)

np_array = np.array([1, 2, 3, 4, 5])
print(type(np_array))
print(np_array)
print(np_array.shape)

# set the dtype attribute to set the data type of all elements.
array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
print(array_2d)
print(type(array_2d))
print(array_2d.shape)

zero_arr = np.zeros((5, 5), dtype=np.float32)
print(type(zero_arr))
print(zero_arr)

ones_arr = np.ones((2, 2), dtype=np.int32)
print(ones_arr)

sixes_arr = np.full((3, 3), 6)
print(sixes_arr)

identity_mat = np.eye(5)
print(identity_mat)

# random values between 0 and 1
random_arr =np.random.random((5, 5))
print(random_arr)

random_arr = np.random.randint(0, 10, (3, 3))
print(random_arr)

# array of evenly spaced values
eq_spaced_arr = np.linspace(1, 100, 50)
print(eq_spaced_arr)

# np array manipulation
arr = np.random.randint(10, 20, 5)
print(arr)

print(arr.transpose())
print(array_2d.transpose())

# reshaping array
array_2d = np.array([[1, 2], [3, 4], [5, 6]])
print(array_2d.reshape(1, 6))