import numpy as np
from icecream import ic
import matplotlib.pyplot as plt
s2_cloudy = np.load('input_opt_batch.npy')
ic(s2_cloudy.shape)
s2_cloudy = np.transpose(s2_cloudy[0, 1:4], (1, 2, 0))
ic(s2_cloudy.shape)


s2 = np.load('output_opt_batch.npy')
ic(s2.shape)
s2 = np.transpose(s2[0, 1:4], (1, 2, 0))
ic(s2.shape)

fig, axs = plt.subplots(2)
fig.suptitle('Vertically stacked subplots')
axs[0].imshow(s2_cloudy)
axs[1].imshow(s2)

#plt.figure()
#plt.imshow(s2_cloudy)
plt.show()