import skimage.io
import json as js
import numpy as np
import matplotlib.pyplot as plt
from colorsys import rgb_to_hls
from colorsys import hls_to_rgb
from sklearn.mixture import GaussianMixture


def convert2hls(palettes):
    """
    Convert RGB to HLS

    Parameters:
        palettes : numpy array has shape (n_palettes, 3) in RGB
    Returns:
        Python list has shape (n_palettes, 3) in HLS
    """
    palettes = palettes.astype(np.float32)
    palettes /= 255.0
    return [rgb_to_hls(pal[0], pal[1], pal[2]) for pal in palettes]


def convert2rgb(palettes):
    """
    Convert HLS to RGB

    Parameters:
        palettes : Python list has shape (n_palettes, 3) in HLS
    Returns:
        numpy array has shape (n_palettes, 3) in RGB
    """
    palettes = np.array([hls_to_rgb(pal[0], pal[1], pal[2]) for pal in palettes])
    palettes *= 255.0
    return palettes.astype(np.uint8)


def sort_palette_by_hls(palettes):
    """
    Sort palettes by Light, Saturation, Hue

    Parameters:
        palettes : Python list has shape (n_palettes, 3) in HLS
    Returns:
        Python list has shape (n_palettes, 3) in HLS
    """
    return sorted(palettes, key=lambda palette: (palette[1],
                                                 palette[2],
                                                 palette[0]))


def generate_palette(filename, n_palettes=5, random_state=42):
    """
    Generate palettes from a given image

    Parameters:
        filename : a path string to an image file or an URL
        n_palettes : number of color palettes you wish to generate
        random_state : GaussianMixture's random_state
    Returns:
        numpy array has shape (n_palettes, 3) in RGB
    """
    # Read the image
    img = skimage.io.imread(filename)
    w, h, n_channels = img.shape
    img = np.reshape(img, (w * h, n_channels))

    # Create a cluster
    cluster = GaussianMixture(n_components=n_palettes, max_iter=100,
                              random_state=random_state).fit(img)
    centers = cluster.means_
    palettes = centers.astype(np.uint8)

    # Convert to HLS
    palettes = convert2hls(palettes)
    # Sort by Light, Saturation, Hue
    palettes = sort_palette_by_hls(palettes)
    # Convert to RGB
    palettes = convert2rgb(palettes)

    return palettes


def visualize_palette(palettes):
    """
    Visualizes the palettes

    Parameters:
        palettes : numpy array has shape (n_palettes, 3) in RGB
    Returns:
        None
    """
    for i, pal in enumerate(palettes):
        plt.subplot(1, len(palettes), i+1)
        plt.title(str(pal.tolist()))
        plt.imshow([[pal]])
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    return None


if __name__ == '__main__':
    filename = 'sample.jpg'
    palettes = generate_palette(filename, n_palettes=5)
    visualize_palette(palettes)

    # To port or transfer palettes to json/dict object
    # simple call palettes.tolist()
    # to convert numpy array to python list
    # palettes.tolist()
