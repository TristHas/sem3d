import sys
import time as t

import cv2
import numpy as np

class Direction:
    def __init__(self, direction=(0, 0), name='invalid'):
        """
        represent a cardinal direction in image coordinates (top left = (0, 0) and bottom right = (1, 1)).
        :param direction: (x, y) for cardinal direction.
        :param name: common name of said direction.
        """
        self.direction = direction
        self.name = name

# 8 defined directions for sgm
N = Direction(direction=(0, -1), name='north')
NE = Direction(direction=(1, -1), name='north-east')
E = Direction(direction=(1, 0), name='east')
SE = Direction(direction=(1, 1), name='south-east')
S = Direction(direction=(0, 1), name='south')
SW = Direction(direction=(-1, 1), name='south-west')
W = Direction(direction=(-1, 0), name='west')
NW = Direction(direction=(-1, -1), name='north-west')

class Paths:
    def __init__(self):
        """
        represent the relation between the directions.
        """
        self.paths = [N, NE, E, SE, S, SW, W, NW]
        self.size = len(self.paths)
        self.effective_paths = [(E,  W), (SE, NW), (S, N), (SW, NE)]


class Parameters:
    def __init__(self, max_disparity=64, P1=5, P2=70, csize=(7, 7), bsize=(3, 3)):
        """
        represent all parameters used in the sgm algorithm.
        :param max_disparity: maximum distance between the same pixel in both images.
        :param P1: penalty for disparity difference = 1
        :param P2: penalty for disparity difference > 1
        :param csize: size of the kernel for the census transform.
        :param bsize: size of the kernel for blurring the images and median filtering.
        """
        self.max_disparity = max_disparity
        self.P1 = P1
        self.P2 = P2
        self.csize = csize
        self.bsize = bsize


def load_images(left_name, right_name, parameters):
    """
    read and blur stereo image pair.
    :param left_name: name of the left image.
    :param right_name: name of the right image.
    :param parameters: structure containing parameters of the algorithm.
    :return: blurred left and right images.
    """
    left = cv2.imread(left_name, 0)
    left = cv2.GaussianBlur(left, parameters.bsize, 0, 0)
    right = cv2.imread(right_name, 0)
    right = cv2.GaussianBlur(right, parameters.bsize, 0, 0)
    return left, right

def get_indices(offset, dim, direction, height):
    """
        for the diagonal directions (SE, SW, NW, NE), return the array of indices for the current slice.
        :param offset: difference with the main diagonal of the cost volume.
        :param dim: number of elements along the path.
        :param direction: current aggregation direction.
        :param height: H of the cost volume.
        :return: arrays for the y (H dimension) and x (W dimension) indices.
    """
    
    y_indices = []
    x_indices = []

    for i in range(0, dim):
        if direction == SE.direction:
            if offset < 0:
                y_indices.append(-offset + i)
                x_indices.append(0 + i)
            else:
                y_indices.append(0 + i)
                x_indices.append(offset + i)

        if direction == SW.direction:
            if offset < 0:
                y_indices.append(height + offset - i)
                x_indices.append(0 + i)
            else:
                y_indices.append(height - i)
                x_indices.append(offset + i)

    return np.array(y_indices), np.array(x_indices)

def aggregate_costs(cost_volume, parameters, paths):
    """
    second step of the sgm algorithm, aggregates matching costs for N possible directions (8 in this case).
    :param cost_volume: array containing the matching costs.
    :param parameters: structure containing parameters of the algorithm.
    :param paths: structure containing all directions in which to aggregate costs.
    :return: H x W x D x N array of matching cost for all defined directions.
    """
    height = cost_volume.shape[0]
    width = cost_volume.shape[1]
    disparities = cost_volume.shape[2]
    start = -(height - 1)
    end = width - 1

    aggregation_volume = np.zeros(shape=(height, width, disparities, paths.size), dtype=np.uint32)

    path_id = 0
    
    for path in paths.effective_paths:
        print('\tProcessing paths {} and {}...'.format(path[0].name, path[1].name), end='')
        sys.stdout.flush()
        dawn = t.time()

        main_aggregation = np.zeros(shape=(height, width, disparities), dtype=np.uint32)
        opposite_aggregation = np.copy(main_aggregation)

        main = path[0]
        if main.direction == S.direction:
            for x in range(0, width):
                south = cost_volume[0:height, x, :]
                north = np.flip(south, axis=0)
                main_aggregation[:, x, :] = get_path_cost(south, 1, parameters)
                opposite_aggregation[:, x, :] = np.flip(get_path_cost(north, 1, parameters), axis=0)

        if main.direction == E.direction:
            for y in range(0, height):
                east = cost_volume[y, 0:width, :]
                west = np.flip(east, axis=0)
                main_aggregation[y, :, :] = get_path_cost(east, 1, parameters)
                opposite_aggregation[y, :, :] = np.flip(get_path_cost(west, 1, parameters), axis=0)

        if main.direction == SE.direction:
            for offset in range(start, end):
                south_east = cost_volume.diagonal(offset=offset).T
                north_west = np.flip(south_east, axis=0)
                dim = south_east.shape[0]
                y_se_idx, x_se_idx = get_indices(offset, dim, SE.direction, None)
                y_nw_idx = np.flip(y_se_idx, axis=0)
                x_nw_idx = np.flip(x_se_idx, axis=0)
                main_aggregation[y_se_idx, x_se_idx, :] = get_path_cost(south_east, 1, parameters)
                opposite_aggregation[y_nw_idx, x_nw_idx, :] = get_path_cost(north_west, 1, parameters)

        if main.direction == SW.direction:
            for offset in range(start, end):
                south_west = np.flipud(cost_volume).diagonal(offset=offset).T
                north_east = np.flip(south_west, axis=0)
                dim = south_west.shape[0]
                y_sw_idx, x_sw_idx = get_indices(offset, dim, SW.direction, height - 1)
                y_ne_idx = np.flip(y_sw_idx, axis=0)
                x_ne_idx = np.flip(x_sw_idx, axis=0)
                main_aggregation[y_sw_idx, x_sw_idx, :] = get_path_cost(south_west, 1, parameters)
                opposite_aggregation[y_ne_idx, x_ne_idx, :] = get_path_cost(north_east, 1, parameters)

        aggregation_volume[:, :, :, path_id] = main_aggregation
        aggregation_volume[:, :, :, path_id + 1] = opposite_aggregation
        path_id = path_id + 2

        dusk = t.time()
        print('\t(done in {:.2f} s)'.format(dusk - dawn))

    return aggregation_volume

def get_path_cost(slice, offset, parameters):
    """
        part of the aggregation step, finds the minimum costs in a D x M slice 
        (where M = the number of pixels in the given direction)
        :param slice: M x D array from the cost volume.
        :param offset: ignore the pixels on the border.
        :param parameters: structure containing parameters of the algorithm.
        :return: M x D array of the minimum costs for a given slice in a given direction.
    """
    other_dim = slice.shape[0]
    disparity_dim = slice.shape[1]

    disparities = [d for d in range(disparity_dim)] * disparity_dim
    disparities = np.array(disparities).reshape(disparity_dim, disparity_dim)

    penalties = np.zeros(shape=(disparity_dim, disparity_dim), dtype=np.uint32)
    penalties[np.abs(disparities - disparities.T) == 1] = parameters.P1
    penalties[np.abs(disparities - disparities.T) > 1] = parameters.P2

    minimum_cost_path = np.zeros(shape=(other_dim, disparity_dim), dtype=np.uint32)
    minimum_cost_path[offset - 1, :] = slice[offset - 1, :]

    for i in range(offset, other_dim):
        previous_cost = minimum_cost_path[i - 1, :]
        current_cost = slice[i, :]
        costs = np.repeat(previous_cost, repeats=disparity_dim, axis=0).reshape(disparity_dim, disparity_dim)
        costs = np.amin(costs + penalties, axis=0)
        minimum_cost_path[i, :] = current_cost + costs - np.amin(previous_cost)
    return minimum_cost_path

def select_disparity(aggregation_volume):
    """
    last step of the sgm algorithm, corresponding to equation 14 followed by winner-takes-all approach.
    :param aggregation_volume: H x W x D x N array of matching cost for all defined directions.
    :return: disparity image.
    """
    volume = np.sum(aggregation_volume, axis=3)
    disparity_map = np.argmin(volume, axis=2)
    return disparity_map


def normalize(volume, parameters):
    """
    transforms values from the range (0, 64) to (0, 255).
    :param volume: n dimension array to normalize.
    :param parameters: structure containing parameters of the algorithm.
    :return: normalized array.
    """
    return 255.0 * volume / parameters.max_disparity

def compute_costs(left, right, parameters, save_images):
    """
    first step of the sgm algorithm, matching cost based on census transform and hamming distance.
    :param left: left image.
    :param right: right image.
    :param parameters: structure containing parameters of the algorithm.
    :param save_images: whether to save census images or not.
    :return: H x W x D array with the matching costs.
    """
    assert left.shape[0] == right.shape[0] and left.shape[1] == right.shape[1], 'left & right must have the same shape.'
    assert parameters.max_disparity > 0, 'maximum disparity must be greater than 0.'

    height = left.shape[0]
    width  = left.shape[1]
    cheight = parameters.csize[0]
    cwidth  = parameters.csize[1]
    y_offset = int(cheight / 2)
    x_offset = int(cwidth / 2)
    disparity = parameters.max_disparity

    left_img_census     = np.zeros(shape=(height, width), dtype=np.uint8)
    right_img_census    = np.zeros(shape=(height, width), dtype=np.uint8)
    left_census_values  = np.zeros(shape=(height, width), dtype=np.uint64)
    right_census_values = np.zeros(shape=(height, width), dtype=np.uint64)

    print('\tComputing left and right census...', end='')
    sys.stdout.flush()
    dawn = t.time()
    # pixels on the border will have no census values
    for y in range(y_offset, height - y_offset):
        for x in range(x_offset, width - x_offset):
            left_census = 0
            # left census transform = from right image
            center_pixel = right[y, x]
            reference = np.full(shape=(cheight, cwidth), fill_value=center_pixel, dtype=np.int64)
            
            image = right[(y - y_offset):(y + y_offset + 1), (x - x_offset):(x + x_offset + 1)]
            comparison = image - reference
            
            for j in range(comparison.shape[0]):
                for i in range(comparison.shape[1]):
                    if (i, j) != (y_offset, x_offset):
                        left_census = left_census << 1
                        if comparison[j, i] < 0:
                            bit = 1
                        else:
                            bit = 0
                        left_census = left_census | bit
            left_img_census[y, x] = np.uint8(left_census)
            left_census_values[y, x] = left_census

            right_census = 0
            # right census transform = from left image
            center_pixel = left[y, x]
            reference = np.full(shape=(cheight, cwidth), fill_value=center_pixel, dtype=np.int64)
            image = left[(y - y_offset):(y + y_offset + 1), (x - x_offset):(x + x_offset + 1)]
            comparison = image - reference
            for j in range(comparison.shape[0]):
                for i in range(comparison.shape[1]):
                    if (i, j) != (y_offset, x_offset):
                        right_census = right_census << 1
                        if comparison[j, i] < 0:
                            bit = 1
                        else:
                            bit = 0
                        right_census = right_census | bit
            right_img_census[y, x] = np.uint8(right_census)
            right_census_values[y, x] = right_census

    dusk = t.time()
    print('\t(done in {:.2f} s)'.format(dusk - dawn))

    if save_images:
        cv2.imwrite('left_census.png', left_img_census)
        cv2.imwrite('right_census.png', right_img_census)

    print('\tComputing cost volume...', end='')
    sys.stdout.flush()
    dawn = t.time()
    cost_volume = np.zeros(shape=(height, width, disparity), dtype=np.uint32)
    rcensus = np.zeros(shape=(height, width), dtype=np.int32)
    for d in range(0, disparity):
        
        rcensus[:, x_offset:(width - d - x_offset)] =\
            right_census_values[:, (x_offset + d):(width - x_offset)]
        
        rcensus[:, (width - d - x_offset):(width - x_offset)] = \
            right_census_values[:, (width - disparity - x_offset):(width - x_offset - disparity + d)]
        
        xor = np.int64(np.bitwise_xor(np.int32(left_census_values), rcensus))
        distance = np.zeros(shape=(height, width), dtype=np.uint32)
        
        while not np.all(xor == 0):
            tmp = xor - 1
            mask = xor != 0
            xor[mask] = np.bitwise_and(xor[mask], tmp[mask])
            distance[mask] = distance[mask] + 1
        cost_volume[:, :, d] = distance

    dusk = t.time()
    print('\t(done in {:.2f} s)'.format(dusk - dawn))

    return cost_volume

def sgm(left, right):
    """
    main function applying the semi-global matching algorithm.
    :return: void.
    """
    output_name = "./data/output"
    disparity = 64
    save_images = False

    parameters = Parameters(max_disparity=disparity, P1=5, P2=70, csize=(7, 7), bsize=(3, 3))
    paths = Paths()

    #print('\nLoading images...')
    left, right = load_images(left, right, parameters)

    print('\nStarting cost computation...')
    cost_volume = compute_costs(left, right, parameters, save_images)
    
    #if save_images:
    #    disparity_map = np.uint8(normalize(np.argmin(cost_volume, axis=2), parameters))
    #    cv2.imwrite('disp_map_cost_volume.png', disparity_map)

    print('\nStarting aggregation computation...')
    aggregation_volume = aggregate_costs(cost_volume, parameters, paths)

    print('\nSelecting best disparities...')
    disparity_map = np.uint8(normalize(select_disparity(aggregation_volume), parameters))
    
    #if save_images:
    #    cv2.imwrite('disp_map_no_post_processing.png', disparity_map)

    print('\nApplying median filter...')
    disparity_map = cv2.medianBlur(disparity_map, parameters.bsize[0])
    #cv2.imwrite(output_name, disparity_map)
    return disparity_map


###
### New code
###

def _pixel_census(image):
    """
    """
    cheight, cwidth = image.shape
    y_offset = int(cheight / 2)
    x_offset = int(cwidth / 2)

    center_pixel = image[y_offset, x_offset]
    reference = np.full(shape=(cheight, cwidth), fill_value=center_pixel, dtype=np.int64)
    comparison = image - reference
    
    census = 0

    for j in range(comparison.shape[0]):
        for i in range(comparison.shape[1]):
            if (i, j) != (y_offset, x_offset):
                census = census << 1
                if comparison[j, i] < 0:
                    bit = 1
                else:
                    bit = 0
                census = census | bit
    return census

def images_census(left, right, height, width, x_offset, y_offset):
    left_census_values  = np.zeros(shape=(height, width), dtype=np.uint64)
    right_census_values = np.zeros(shape=(height, width), dtype=np.uint64)

    # pixels on the border will have no census values
    for y in range(y_offset, height - y_offset):
        for x in range(x_offset, width - x_offset):
            image = right[(y - y_offset):(y + y_offset + 1), (x - x_offset):(x + x_offset + 1)]
            left_census = _pixel_census(image)
            left_census_values[y, x] = left_census
            
            image =  left[(y - y_offset):(y + y_offset + 1), (x - x_offset):(x + x_offset + 1)]
            right_census = _pixel_census(image)
            right_census_values[y, x] = right_census
    return (left_census_values,
           right_census_values)

def compare_census(left_census_values, right_census_values, height, width, disparity, x_offset):
    """
    """
    cost_volume = np.zeros(shape=(height, width, disparity), dtype=np.uint32)
    rcensus = np.zeros(shape=(height, width), dtype=np.int32)
    
    for d in range(0, disparity):
        
        rcensus[:, x_offset:(width - d - x_offset)] =\
            right_census_values[:, (x_offset + d):(width - x_offset)]
        
        rcensus[:, (width - d - x_offset):(width - x_offset)] = \
            right_census_values[:, (width - disparity - x_offset):(width - x_offset - disparity + d)]
        
        xor = np.int64(np.bitwise_xor(np.int32(left_census_values), rcensus))
        distance = np.zeros(shape=(height, width), dtype=np.uint32)
        
        while not np.all(xor == 0):
            tmp = xor - 1
            mask = xor != 0
            xor[mask] = np.bitwise_and(xor[mask], tmp[mask])
            distance[mask] = distance[mask] + 1
        cost_volume[:, :, d] = distance

    return cost_volume

def census_costs(left, right, parameters):
    """
    first step of the sgm algorithm, matching cost based on census transform and hamming distance.
    :param left: left image.
    :param right: right image.
    :param parameters: structure containing parameters of the algorithm.
    :return: H x W x D array with the matching costs.
    """
    assert left.shape[0] == right.shape[0] and left.shape[1] == right.shape[1], 'left & right must have the same shape.'
    assert parameters.max_disparity > 0, 'maximum disparity must be greater than 0.'

    height = left.shape[0]
    width  = left.shape[1]
    cheight = parameters.csize[0]
    cwidth  = parameters.csize[1]
    y_offset = int(cheight / 2)
    x_offset = int(cwidth / 2)
    disparity = parameters.max_disparity

    print('\tComputing left and right census...', end='')
    sys.stdout.flush()
    dawn = t.time()
    left_feat, right_feat = images_census(left, right, height, width, x_offset, y_offset)
    dusk = t.time()
    print('\t(done in {:.2f} s)'.format(dusk - dawn))

    print('\tComputing cost volume...', end='')
    sys.stdout.flush()
    dawn = t.time()
    cost_volume = compare_census(left_feat, right_feat, height, width, disparity, x_offset)
    dusk = t.time()
    print('\t(done in {:.2f} s)'.format(dusk - dawn))

    return cost_volume

def _pixel_correl(image):
    """
    """
    x = image.flatten()
    return (x-x.mean()) / x.std()

def images_correl(left, right, height, width, x_offset, y_offset):
    left_census_values  = np.zeros(shape=(height, width, ((x_offset*2+1) * (y_offset*2+1))), dtype=left.dtype)
    right_census_values = np.zeros(shape=(height, width, ((x_offset*2+1) * (y_offset*2+1))), dtype=right.dtype)

    for y in range(y_offset, height - y_offset):
        for x in range(x_offset, width - x_offset):
            image = right[(y - y_offset):(y + y_offset + 1), (x - x_offset):(x + x_offset + 1)]
            left_census = _pixel_correl(image)
            left_census_values[y, x] = left_census
            
            image =  left[(y - y_offset):(y + y_offset + 1), (x - x_offset):(x + x_offset + 1)]
            right_census = _pixel_correl(image)
            right_census_values[y, x] = right_census
    return (left_census_values,
           right_census_values)

def compare_correl(left_census_values, right_census_values, height, width, disparity, x_offset):
    """
    """
    cost_volume = np.zeros(shape=(height, width, disparity), dtype=np.uint32)
    rcensus = np.zeros(shape=right_census_values.shape, dtype=right_census_values.dtype)
    
    for d in range(0, disparity):
        rcensus[:, x_offset:(width - d - x_offset)] =\
            right_census_values[:, (x_offset + d):(width - x_offset)]
        
        rcensus[:, (width - d - x_offset):(width - x_offset)] = \
            right_census_values[:, (width - disparity - x_offset):(width - x_offset - disparity + d)]
        
        distance =  (rcensus*left_census_values).sum(-1)
        cost_volume[:, :, d] = distance

    return cost_volume

def correl_costs(left, right, parameters):
    height = left.shape[0]
    width  = left.shape[1]
    cheight = parameters.csize[0]
    cwidth  = parameters.csize[1]
    y_offset = int(cheight / 2)
    x_offset = int(cwidth / 2)
    disparity = parameters.max_disparity

    left_feat, right_feat =  images_correl(left, right, height, width, x_offset, y_offset)
    return compare_correl(left_feat, right_feat, height, width, disparity, x_offset)

def test_legacy():
    cost_volume = census_costs(patch_left, patch_right, parameters)
    old_cost_volume = compute_costs(patch_left, patch_right, parameters, False)
    assert (cost_volume - old_cost_volume).max()