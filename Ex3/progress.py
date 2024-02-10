
def blend_pyramids_based_on_level(pyramid1, pyramid2, max_level_1):
    blended_pyramid = []
    i = 0
    for i in range(max_level_1):
        blended_pyramid.append({'L': pyramid1[i]['L']})
    for i in range(max_level_1, len(pyramid1)-1):
        blended_pyramid.append({'L': pyramid2[i]['L']})
    blended_pyramid.append({'G': pyramid2[i+1]['G']})
    return blended_pyramid



img_1 = get_gray_img_mat('./assets/milel2.png')
img_2 = get_gray_img_mat('./assets/tutit2.png')
pyramid1 = create_pyramid(img_1)