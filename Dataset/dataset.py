from matplotlib.widgets import Slider
import numpy as np
import matplotlib.pyplot as plt
import glob
from pydicom import dcmread, examples
from pydicom.pixels import pixel_array

path_folder = r"C:\Users\mthou\Documents\Projet phys med\Dataset\lidc_idri\LIDC-IDRI-0002\98329\04919"
files = glob.glob(path_folder + r"\*.dcm")

dicom_files = []
for f in files:
    ds = dcmread(f, stop_before_pixels=True)
    z_position = float(ds.ImagePositionPatient[2])
    dicom_files.append((z_position, f))

dicom_files.sort()

image = []
for _, f in dicom_files:
    ds = dcmread(f)
    arrayTest = pixel_array(ds)
    normal_arrayTest = (arrayTest - np.min(arrayTest)) / (np.max(arrayTest) - np.min(arrayTest))
    image.append(normal_arrayTest)


fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
im = ax.imshow(image[0], cmap='gray')

ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_slider, 'Image', 0, len(image)-1, valinit=0, valstep=1)

def update(val):
    im.set_data(image[int(slider.val)])
    fig.canvas.draw_idle()

slider.on_changed(update)    
plt.show()

