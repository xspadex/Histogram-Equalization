import WGIF
import utils
import basic_he
import dphe
import bhe2pl
import BBHE
import DSIHE
import MMBEBHE
import CLAHE

DRAW_HISTOGRAM = False
DRAW_PICS = False
DRAW_3D_MAP = False
METHOD = ["Histogram_Equalize", "WGIF_Based_Enhance",
          "BBHE", "DSIHE", "MMBEBHE", "DPHE", "BHE2PL", "CLAHE"][1]

RESTORT_COLOR = True


pic_list = [
    "sample01.jpg",
    "sample02.jpeg",
    "sample03.jpeg",
    "sample04.jpeg",
    "sample05.jpeg",
    "sample06.jpg",
    "sample07.jpg",
    "sample08.jpg",
]

for pic in pic_list:
    img = utils.read_img("pics/" + pic)
    gray_img = utils.RGB_to_gray(img)
    intensity = utils.RGB_to_intensity(img)
    if METHOD == "WGIF_Based_Enhance":
        enhanced_intensity = WGIF.wgif_based_enhance(intensity)
        enhanced_gray_img = utils.intensity_to_gray(enhanced_intensity)
    elif METHOD == "Histogram_Equalize":
        enhanced_gray_img = basic_he.histogram_equalize(gray_img)
        enhanced_intensity = utils.gray_to_intensity(enhanced_gray_img)
    elif METHOD == "DPHE":
        enhanced_gray_img = dphe.dph_equalization(gray_img)
        enhanced_intensity = utils.gray_to_intensity(enhanced_gray_img)
    elif METHOD == "BHE2PL":
        enhanced_gray_img = bhe2pl.bi_histogram_equalization(gray_img)
        enhanced_intensity = utils.gray_to_intensity(enhanced_gray_img)
    elif METHOD == "BBHE":
        enhanced_gray_img = BBHE.BBHE(gray_img)
        enhanced_intensity = utils.gray_to_intensity(enhanced_gray_img)
    elif METHOD == "DSIHE":
        enhanced_gray_img = DSIHE.DSIHE(gray_img)
        enhanced_intensity = utils.gray_to_intensity(enhanced_gray_img)
    elif METHOD == "MMBEBHE":
        enhanced_gray_img = MMBEBHE.MMBEBHE(gray_img)
        enhanced_intensity = utils.gray_to_intensity(enhanced_gray_img)
    elif METHOD == "CLAHE":
        enhanced_gray_img = CLAHE.CLAHE(gray_img, 2.0, (8, 8))
        enhanced_intensity = utils.gray_to_intensity(enhanced_gray_img)
    if DRAW_3D_MAP:
        utils.draw_2_matrixs_in_one([gray_img, enhanced_gray_img])
        
    if DRAW_HISTOGRAM:
        hist, bins = basic_he.get_histogram(gray_img)
        new_hist, new_bins = basic_he.get_histogram(enhanced_gray_img)
        utils.draw_histograms_in_one(hist, bins, new_hist, new_bins)

    if RESTORT_COLOR:
        enhanced_img = utils.restort_color(img, intensity, enhanced_intensity)
    else:
        img = gray_img
        enhanced_img = enhanced_gray_img

    if DRAW_PICS:
        utils.show_images_concat(img, enhanced_img)
    
    utils.write_img("results/" + pic.replace(".jpeg", ".jpg"), enhanced_img)
