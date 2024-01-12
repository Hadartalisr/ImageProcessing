import os
import sys

import mediapy as media
import numpy as np
from PIL import Image

NUM_OF_GRAY_LEVELS = 256
ALL_GREYSCALE_HISTOGRAM_BINS = np.arange(NUM_OF_GRAY_LEVELS + 1)
GREYSCALE_HISTOGRAM_BINS_VALUES = np.arange(NUM_OF_GRAY_LEVELS)


# Skeleton function - DO NOT CHANGE SIGNATURE!!!
def main(video_path, video_type):
    """
    Main entry point for ex1
    :param video_path: path to video file
    :param video_type: category of the video (either 1 or 2)
    :return: a tuple of integers representing the frame number for which the scene cut was detected (i.e. the last frame index of the first scene and the first frame index of the second scene)
    """
    if video_type == 1:
        return process_video_type_1(video_path)
    elif video_type == 2:
        return process_video_type_2(video_path)
    else:
        raise ValueError("Invalid video type")


def internal_main():
    """
    Internal main entry point for the program.
    """
    try:
        # Check if arguments are provided from the terminal
        if len(sys.argv) != 3:
            print("Usage: python ex1.py <video_path> <video_type>")
            sys.exit(1)

        video_path = sys.argv[1]
        video_type = int(sys.argv[2])

        result = main(video_path, video_type)
        print(f"Scene cut detected between frames: {result}")
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


def process_video_type_1(video_path):
    video = get_greyscale_video_by_file_path(video_path)
    video_histogram = get_video_histogram(video)
    video_cumsum_histogram = cumsum_along_y(video_histogram)
    cumsum_histogram_distance = sum_diff_consecutive_rows(video_cumsum_histogram)
    frame_with_max_distance = np.argmax(cumsum_histogram_distance)
    return frame_with_max_distance, (frame_with_max_distance + 1)


def process_video_type_2(video_path):
    video = get_greyscale_video_by_file_path(video_path)
    values_arr = get_video_histogram_edges(video)
    quantized_video = quantize_video(video, values_arr)
    video_histogram = get_video_histogram(quantized_video)
    video_cumsum_histogram = cumsum_along_y(video_histogram)
    cumsum_histogram_distance = sum_diff_consecutive_rows(video_cumsum_histogram)
    frame_with_max_distance = np.argmax(cumsum_histogram_distance)
    return frame_with_max_distance, (frame_with_max_distance + 1)

# video4_histogram = cumsum_along_y(video_histogram)
# cumsum_histogram_distance = sum_diff_consecutive_rows(video4_histogram)
# plt.plot(cumsum_histogram_distance)
# print(np.argmax(cumsum_histogram_distance))



def get_video_by_file_path(path: str):
    """

    :rtype: mediapy_video
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return media.read_video(path)


def get_grayscale_video(mediapy_video) -> np.ndarray:
    num_frames, height, width, channels = mediapy_video.shape
    greyscale_video = np.empty((num_frames, height, width), dtype=np.uint8)

    for i in range(num_frames):
        greyscale_video[i] = get_greyscale_array_from_frame(mediapy_video[i])

    return greyscale_video


def get_greyscale_array_from_frame(mediapy_frame) -> np.ndarray:
    return np.array(Image.fromarray(np.array(mediapy_frame)).convert('L'))


def get_greyscale_video_by_file_path(path):
    return get_grayscale_video(get_video_by_file_path(path))


def get_frame_histogram_with_defined_bins(frame: np.ndarray, bins: np.ndarray = ALL_GREYSCALE_HISTOGRAM_BINS) -> np.ndarray:
    hist, _ = np.histogram(frame, bins=bins)
    return hist


def get_video_histogram(video) -> np.ndarray:
    num_frames, _, _ = video.shape
    video_hist = np.zeros((num_frames, NUM_OF_GRAY_LEVELS))
    for i in range(num_frames):
        video_hist[i] = get_frame_histogram_with_defined_bins(video[i])
    return video_hist


def cumsum(histogram):
    return np.cumsum(histogram)


def cumsum_along_y(matrix):
    """
    Calculate the cumulative sum along the y-axis for each x in a 2D array.

    Parameters:
    - matrix (np.ndarray): Input 2D array.

    Returns:
    - cumsum_result (np.ndarray): 2D array containing cumulative sum for each x along the y-axis.
    """
    # Calculate cumulative sum along the y-axis
    cumsum_result = np.cumsum(matrix, axis=1)

    return cumsum_result


def sum_diff_consecutive_rows(matrix):
    """
    Calculate the sum of differences between consecutive rows in a 2D matrix.

    Parameters:
    - matrix (np.ndarray): Input 2D matrix.

    Returns:
    - sum_diff (np.ndarray): 1D array containing the sum of differences for each column.
    """
    # Calculate differences between consecutive rows
    signed_matrix = matrix.astype(np.int64)
    # Calculate absolute differences between consecutive rows
    abs_diff = np.abs(np.diff(signed_matrix, axis=0))
    # Sum absolute differences along the axis representing rows
    sum_abs_diff = np.sum(abs_diff, axis=1)
    return sum_abs_diff


def get_video_histogram_edges(video, num_of_bins=9, range=(0,255)) -> np.ndarray:
    _, edges = np.histogram(video, bins=num_of_bins, range=range)
    return edges.astype(np.uint8)


def map_to_closest_values(matrix, values):
    """
    Map each pixel value in a 2D matrix to the closest value in the given array.

    Parameters:
    - matrix (np.ndarray): Input 2D matrix of integers.
    - values (np.ndarray): Array of integers to map to.

    Returns:
    - mapped_matrix (np.ndarray): Matrix with each pixel value mapped to the closest value in the array.
    """
    # Flatten the matrix and values arrays for easier computation
    flat_matrix = matrix.flatten()
    flat_values = values.flatten()

    # Find the index of the closest value in values for each pixel in the matrix
    closest_indices = np.argmin(np.abs(flat_matrix[:, np.newaxis] - flat_values), axis=1)

    # Map each pixel value to the closest value in values
    mapped_matrix = flat_values[closest_indices].reshape(matrix.shape)

    return mapped_matrix


def quantize_video(video, values_arr):
    quantized_video = np.zeros(video.shape).astype(np.uint8)
    for i in range(video.shape[0]):
        image = map_to_closest_values(video[i], values_arr)
        quantized_video[i] = image
    return quantized_video


if __name__ == "__main__":
    internal_main()
