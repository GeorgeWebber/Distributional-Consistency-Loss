import brainweb as bw
import numpy as np
from skimage.transform import resize

"""BrainWeb phantom generation.
The BrainWeb package was inspired by https://brainweb.bic.mni.mcgill.ca/brainweb/anatomic_normal_20.html"""

def get_brainweb_phantom(n = 256):
	 # Make *** PET FDG *** from slice 160 of the 3D tissue map
	tissue = np.pad(bw.load_file(bw.get_files()[0]), ((0,0), (12,12), (48,48))).astype('f4')
	white_matter = (tissue == 48);  grey_matter = (tissue == 32);  skin = (tissue == 96)
	pet = np.zeros_like(tissue); pet[white_matter]=32; pet[grey_matter]=96; pet[skin]=16
	true = np.flipud(resize(pet[160,:,:], (n,n)))
	return true / 200.0