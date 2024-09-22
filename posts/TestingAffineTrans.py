

import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
from skimage.transform import SimilarityTransform
from skimage.transform import warp
from skimage.transform import PiecewiseAffineTransform
from skimage.transform import AffineTransform
from skimage.io import imread
import io
from pathlib import Path
from skimage.transform import resize, rescale


tform_shift = AffineTransform(rotation=np.pi/2.0)
homog_trans_matrix=tform_shift.params

print(homog_trans_matrix.shape)

prod=homog_trans_matrix@np.array([1.0,2.0,3.0]).T
print(prod)



old_coordinates=[[0,0,1.0],[0,1,1.0],[0,2,1.0],[0,3,1.0],[0,4,1.0]]
old_coordinates=np.array(old_coordinates)

new_coordinates=old_coordinates.copy()

for i in range(old_coordinates.shape[0]):

    homog_vec=np.array(old_coordinates[i,:])

    new_coordinates[i,:]=homog_trans_matrix@homog_vec



fig,ax=plt.subplots()
ax.plot()
ax.plot(old_coordinates[:,0],old_coordinates[:,1],'r*')
ax.plot(new_coordinates[:,0],new_coordinates[:,1],'k*')

ax.grid(True)
plt.show()


