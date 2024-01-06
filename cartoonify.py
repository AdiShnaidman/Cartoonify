##############################################################################
# FILE: cartoonify.py
# WRITERS: Adi Shnaidman, Adishnaidman
# EXERCISE: Intro2cs2 ex6 2021-2022
# STUDENTS I HAVE DISCUSSED WITH: AVRAHAM MEISEL, DANIEL OHANA, AMIT FARKASH
# DESCRIPTION: Cartoonify task number 6, Digital image Processing
##############################################################################

##############################################################################
#                                   Imports                                  #
##############################################################################
from ex6_helper import *
import ex6_helper
from typing import Optional
import copy
import math
import sys


def separate_channels(image: ColoredImage) -> List[List[List[int]]]:
    """turns [x->y->rgb] to [rgb->y->x] -> seperates channels"""
    m = int(len(image[0][0]))
    colors_output = []

    for channel in range(m):
        colors_output.append([])
        for i in range(len(image)):
            colors_output[channel].append([])
            for j in range(len(image[i])):
                colors_output[channel][i].append([])
                colors_output[channel][i][j] = (image[i][j][channel])

    return colors_output


def combine_channels(channels: List[List[List[int]]]) -> ColoredImage:
    """turns [rgb->y->x] to [x->y->rgb] -> combines channels"""
    combination = []
    rgb = []
    columns = []

    for i in range(len(channels[0])):
        for j in range(len(channels[0][0])):
            for k in range(0, len(channels), 1):
                rgb.append(channels[k][i][j])
            columns.append(rgb)
            rgb = []
        combination.append(columns)
        columns = []

    return combination


def RGB2grayscale(colored_image: ColoredImage) -> SingleChannelImage:
    """turns rgb to grayscaling (using rounding up or down)"""
    gray_scale = []
    i = 0
    j = 0
    columns = []

    while i < len(colored_image):
        while j < len(colored_image[i]):
            rgb = 0.299 * colored_image[i][j][0] + 0.587 * colored_image[i][j][1] + 0.114 * colored_image[i][j][2]
            rgb = rounding(rgb)
            j += 1
            columns.append(rgb)
        i += 1
        j = 0

        gray_scale.append(columns)
        columns = []
    return gray_scale


def rounding(rgb):
    """rounds every given rgb to nearest ceil/floor"""
    rgb *= 10
    if rgb % 10 >= 5:
        rgb /= 10
        rgb = math.ceil(rgb)
        return rgb
    else:
        rgb /= 10
        rgb = math.floor(rgb)
        return rgb


def blur_kernel(size: int) -> Kernel:
    """setting blur kernel"""
    calc_size = 1 / (size ** 2)
    return [[calc_size for j in range(size)] for i in range(size)]


def sum_edges(image, block_size, h, w, height, width, kernel):
    """creates average from sum of close pix"""
    summer = 0
    devided_block_size = block_size // 2

    for b1 in range(block_size):
        for b2 in range(block_size):
            calc_for_height = h + b1 - devided_block_size
            calc_for_width = w + b2 - devided_block_size
            if 0 <= calc_for_height < height and 0 <= calc_for_width < width:
                summer += kernel[b1][b2] * image[calc_for_height][calc_for_width]
            else:
                summer += kernel[b1][b2] * image[h][w]
    if summer < 0:
        summer = 0
    if summer > 255:
        summer = 255

    return summer


def apply_kernel(image: SingleChannelImage, kernel: Kernel) -> SingleChannelImage:
    """gets image and kernel and returns applied kernel"""
    height = len(image)
    width = len(image[0])
    block_size = len(kernel)
    image_to_be_returned = []
    new_image = copy.deepcopy(image)
    for h in range(height):
        image_to_be_returned.append([])
        for w in range(width):
            new_val = round(sum_edges(image, block_size, h, w, height, width, kernel))
            new_image[h][w] = new_val
    return new_image


def bilinear_interpolation(image: SingleChannelImage, y: float, x: float) -> int:
    """returns pixel's new value"""
    if x > len(image[0]) - 1:
        x = len(image[0]) - 1
    if y > len(image) - 1:
        y = len(image) - 1

    a = image[math.floor(y)][math.floor(x)]
    b = image[math.ceil(y)][math.floor(x)]
    c = image[math.floor(y)][math.ceil(x)]
    d = image[math.ceil(y)][math.ceil(x)]

    dis_x = x - math.floor(x)
    dis_y = y - math.floor(y)

    overall_pixel = a * (1 - dis_x) * (1 - dis_y) + b * dis_y * (1 - dis_x) + c * dis_x * (1 - dis_y) + d * (
            dis_x * dis_y)

    return rounding(overall_pixel)


def resize(image: SingleChannelImage, new_height: int, new_width: int) -> SingleChannelImage:
    """risizing pixels with new height and new width"""
    y = (len(image[0]) - 1) / (new_width - 1)
    x = (len(image) - 1) / (new_height - 1)

    r = []
    c = []

    for i in range(new_height):
        for j in range(new_width):
            pix = bilinear_interpolation(image, x * i, y * j)
            c.append(pix)
        r.append(c)
        c = []
    return r


def scale_down_colored_image(image: ColoredImage, max_size: int) -> Optional[ColoredImage]:
    """resizes colored image proportionally"""
    r, c = len(image), len(image[0])
    if len(image) <= max_size and len(image[0]) <= max_size:
        return None

    # setting proportions
    if r > c:
        rows_for_scaling = max_size
        cols_for_scaling = int(round(max_size * c / r))
    else:
        cols_for_scaling = max_size
        rows_for_scaling = int(round(max_size * r / c))
    if cols_for_scaling == 0:
        cols_for_scaling = 1
    if rows_for_scaling == 0:
        rows_for_scaling = 1

    # resizing
    lst_for_resized_n_seperated = []
    separated_image = separate_channels(image)
    for c in range(len(separated_image)):
        lst_for_resized_n_seperated.append(resize(separated_image[c], rows_for_scaling, cols_for_scaling))
    resized_n_combined = combine_channels(lst_for_resized_n_seperated)
    return resized_n_combined


def rotate_90(image: Image, direction: str) -> Image:
    """rotates images 90 degrees right and 90 degrees left"""
    column_r = []
    column_l = []
    rows = []

    if direction == 'L':
        for i in range(len(image[0]) - 1, -1, -1):
            for j in range(len(image)):
                column_l.append(image[j][i])
            rows.append(column_l)
            column_l = []
        return rows

    if direction == 'R':
        for i in range(len(image[0])):
            for j in range(len(image) - 1, -1, -1):
                column_r.append(image[j][i])
            rows.append(column_r)
            column_r = []
        return rows


def get_edges(image: SingleChannelImage, blur_size: int, block_size: int, c: int) -> SingleChannelImage:
    """emphasizes edges"""
    height = len(image)
    width = len(image[0])
    blurred_kernel = blur_kernel(blur_size)
    blurred_image_new = apply_kernel(image, blurred_kernel)
    dc_new_image = copy.deepcopy(blurred_image_new)
    applied_blurred_image = apply_kernel(blurred_image_new, blur_kernel(block_size))
    for i in range(height):
        for j in range(width):
            if blurred_image_new[i][j] < applied_blurred_image[i][j] - c:
                dc_new_image[i][j] = 0
            else:
                dc_new_image[i][j] = 255
    return dc_new_image


def quantize(image: SingleChannelImage, N: int) -> SingleChannelImage:
    """adjustment black&white image to specific colors"""
    c = []
    r = []
    for i in range(len(image)):
        for j in range(len(image[i])):
            qimg = round(math.floor(image[i][j] * N / 256) * 255 / (N - 1))
            r.append(qimg)
        c.append(r)
        r = []

    return c


def quantize_colored_image(image: ColoredImage, N: int) -> ColoredImage:
    """adjustment fully colored image to specific color"""
    seperated = separate_channels(image)
    still_seperated = []

    for i in range(len(seperated)):
        still_seperated.append(quantize(seperated[i], N))
    final = combine_channels(still_seperated)

    return final


def add_mask(image1: Image, image2: Image, mask: List[List[float]]) -> Image:
    """takes two pictures and combines them together"""
    if type(image1[0][0]) == list:
        return colored_add_mask(image1, image2, mask)
    i, j = 0, 0
    final_lst = []
    while i < len(image1):
        outer_pix = []
        while j < len(image1[i]):
            inner_pix = round(image1[i][j] * mask[i][j] + image2[i][j] * (1 - mask[i][j]))
            outer_pix.append(inner_pix)
            j += 1
        final_lst.append(outer_pix)
        j = 0
        i += 1

    return final_lst


def colored_add_mask(image1, image2, mask):
    """masks colored pics"""
    final_lst = copy.deepcopy(image1)
    for i in range(len(image1)):
        for j in range(len(image1[i])):
            for k in range(len(image1[i][j])):
                pix = round(image1[i][j][k] * mask[i][j] + image2[i][j][k] * (1 - mask[i][j]))
                final_lst[i][j][k] = pix

    return final_lst


def mask_for_image(image):
    """creates masks for image, leaves black edges in"""
    dc_of_image = copy.deepcopy(image)
    for i in range(len(image)):
        for j in range(len(image[i])):
            if image[i][j] == 0:
                dc_of_image[i][j] = 0
            if image[i][j] == 255:
                dc_of_image[i][j] = 1

    return dc_of_image


def cartoonify(image: ColoredImage, blur_size: int, th_block_size: int,
               th_c: int, quant_num_shades: int) -> ColoredImage:
    """makes picture cartooned using previous functions"""
    image_edge = get_edges(RGB2grayscale(image), blur_size, th_block_size, th_c)
    image_quan = quantize_colored_image(image, quant_num_shades)
    edge = copy.deepcopy(image_edge)
    for i in range(len(image_edge)):
        for j in range(len(image_edge[i])):
            edge[i][j] = [image_edge[i][j], image_edge[i][j], image_edge[i][j]]
    cartooned = add_mask(image_quan, edge, mask_for_image(image_edge))

    return cartooned


if __name__ == '__main__':
    """runs the whole image processing program"""
    if len(sys.argv) != 8:
        print("number of args is wrong")
        sys.exit()
    image_source = sys.argv[1]
    cartoon_dest = sys.argv[2]
    max_im_size = int(sys.argv[3])
    blur_size = int(sys.argv[4])
    th_block_size = int(sys.argv[5])
    th_c = int(sys.argv[6])
    quant_num_shades = int(sys.argv[7])
    image = ex6_helper.load_image(image_source)
    if scale_down_colored_image(image, max_im_size) is None:
        cartooned = cartoonify(image, blur_size, th_block_size, th_c, quant_num_shades)
    else:
        resized_pic_for_cartoonifying = scale_down_colored_image(image, max_im_size)
        cartooned = cartoonify(resized_pic_for_cartoonifying, blur_size, th_block_size, th_c, quant_num_shades)
    ex6_helper.save_image(cartooned, cartoon_dest)
