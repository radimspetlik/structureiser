from PIL import Image
import torch
import numpy as np

# parse image made up of homogenous subimages
def parse_img(im_combined, rows, cols):
    width, height = im_combined.size
    
    assert(width % cols == 0)
    assert(height % rows == 0)

    subimg_width = width // cols
    subimg_height = height // rows
    # crop makes lazy images - left upper, right lower
    row_list = []
    for r in range(rows):
        col_list = []
        for c in range(cols):
            area = (subimg_width * c, subimg_height * r, subimg_width * (c + 1), subimg_height * (r + 1))
            subimg = im_combined.crop(area)
            col_list.append(subimg)
        row_list.append(col_list)
    return row_list

def pack_img(array_like_of_images, rows, cols):
    size = None
    array = [np.array(x) for x in array_like_of_images]
    for img in array:
        if size is None:
            size = img.shape
        if img.shape != size:
            raise ValueError("All images must be the same shape")
    img_combined = np.empty((rows * size[0], cols * size[1], size[2]), dtype=array[0].dtype)
    img_ptr = 0
    for i in range(rows):
        for j in range(cols):
            if img_ptr >= len(array):
                break
            img_combined[i*size[0]:(i+1)*size[0], j*size[1]:(j+1)*size[1], :] = array[img_ptr]
            img_ptr += 1
    return img_combined

# assume C, H, W and tensor
def cut_patch(chw_tensor, midpoint, size):
    hs = size // 2
    hn = max(0, midpoint[0] - hs)
    hx = min(midpoint[0] + hs, chw_tensor.size()[1] - 1)
    xn = max(0, midpoint[1] - hs)
    xx = min(midpoint[1] + hs, chw_tensor.size()[2] - 1)
    
    p = chw_tensor[:, hn:hx, xn:xx]
    if p.shape[1] != size or p.shape[2] != size:
        r = torch.zeros((chw_tensor.shape[0], size, size))
        r[:, 0:p.shape[1], 0:p.shape[2]] = p
        p = r
    return p

def cut_patches(nchw_tensor, midpoints, size):
    # assume midpoints are valid indices that dont go outside of the (image + size)
    assert nchw_tensor.shape[0] == 1, "Cut patches is not implemented for batches"

    # Can be done using 2x unfold(..), but that will cut out all the patches, whereas I expect
    # that the number of patches will be relatively small
    
    
    half = size // 2
    # the naive way
    ys, xs = midpoints[0], midpoints[1]
    num_patches = len(ys)
    output = torch.empty(num_patches, nchw_tensor.shape[1], size, size).to(nchw_tensor.device)
    
    for h, w, i in zip(ys, xs, range(num_patches)):
        output[i, :, :, :] = nchw_tensor[:, :, int(h-half):int(h+half), int(w-half):int(w+half)]
    return output

if __name__ == "__main__":
    t = torch.rand((1, 3, 64, 64))
    midpoints = torch.Tensor([[13, 52], [32, 24]])
    print(cut_patches(t, midpoints, 5).shape)