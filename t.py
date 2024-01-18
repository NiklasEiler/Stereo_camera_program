import numpy as np

# Assuming binary_image is your binary image (numpy array)
binary_image = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])

# Get coordinates with value 1
coordinates_with_1 = np.array(np.where(binary_image == 1)).T

print("Coordinates with value 1:")
print(coordinates_with_1)