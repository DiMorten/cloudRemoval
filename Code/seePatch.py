import numpy as np
from icecream import ic
import matplotlib.pyplot as plt
s2_cloudy = np.load('input_opt_batch.npy')
ic(s2_cloudy.shape)
s2_cloudy = np.transpose(s2_cloudy[0, 1:4], (1, 2, 0))
ic(s2_cloudy.shape)

plt.figure()
plt.imshow(s2_cloudy)
plt.show()