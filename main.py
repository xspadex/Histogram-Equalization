from basic_he import *

DRAW_HISTOGRAM = False
DRAW_PICS = False
DRAW_3D_MAP = True

pic_list = [
    "./pics/sample01.jpg",
    "./pics/sample02.jpeg",
    "./pics/sample03.jpeg",
    "./pics/sample04.jpeg",
    "./pics/sample05.jpeg",
    "./pics/sample06.jpg",
    "./pics/sample07.jpg",
    "./pics/sample08.jpg",
]

for pic in pic_list:
    img = read_img_gray(pic)

    hed_img = histogram_equalize(img, need_draw_histogram=DRAW_HISTOGRAM)
    if DRAW_3D_MAP:
        draw_2_matrixs_in_one([img, hed_img])
    if DRAW_PICS:
        show_images_concat(img, hed_img)

