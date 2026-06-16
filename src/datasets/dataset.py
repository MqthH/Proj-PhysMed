from matplotlib.widgets import Slider
import numpy as np
import matplotlib.pyplot as plt
import glob
from pydicom import dcmread, examples
from pydicom.pixels import pixel_array


class Dataset:
    def __init__(self, strict=True):
        self.strict = strict

    def analyze(self, file_path):
        with open(file_path, "rb") as dcm_files:
            return self._process_file(dcm_files)
        
    def _process_file(self, dcm_files):
        files = glob.glob(dcm_files + r"\*.dcm")


        ct_files = []
        for f in files:
            ds_ct = dcmread(f, stop_before_pixels=True)
            z_position = float(ds_ct.ImagePositionPatient[2])
            ct_files.append((z_position, f))

        ct_files.sort()


        image_ct = []
        for _, f in ct_files:
            ds = dcmread(f)
            arr_ct = pixel_array(ds)
            normal_arr_ct = (arr_ct - np.min(arr_ct)) / (np.max(arr_ct) - np.min(arr_ct))
            image_ct.append(normal_arr_ct)


        ds_seg = dcmread(r"C:\Users\mthou\Documents\Projet phys med\Dataset\Seg - Patient 0002\LIDC-IDRI-0002\98329\05274\a20cbc44-a7a4-49dc-8c37-5a0a2bcebf13.dcm")
        arr_seg = pixel_array(ds_seg)


        pos_seg = []
        for frame in ds_seg.PerFrameFunctionalGroupsSequence:
            z = float(frame.PlanePositionSequence[0].ImagePositionPatient[2])
            pos_seg.append(z)


        def trouver_ct_matching(z_seg, ct_files):
            distances = [abs(z_ct - z_seg) for z_ct, _ in ct_files]
            return np.argmin(distances)


        image_matching = []
        for z in pos_seg:
            image_matching.append(trouver_ct_matching(z, ct_files))


        """ 
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        im = ax.imshow(image_ct[image_matching[0]], cmap='gray')

        ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
        slider = Slider(ax_slider, 'Image', 0, len(image_matching)-1, valinit=0, valstep=1)

        def update(val):
            im.set_data(image_ct[image_matching[int(slider.val)]])
            fig.canvas.draw_idle()

        slider.on_changed(update)    
        plt.show()
        """
        







