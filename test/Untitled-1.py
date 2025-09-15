import marimo

__generated_with = "0.15.3"
app = marimo.App()


@app.cell
def _():
    import cv2 
    import numpy as np
    import matplotlib.pyplot as plt
    return np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        https://thejacksonlaboratory.github.io/image-processing-ia/instructor/01-introduction.html

        https://scikit-image.org/

        """
    )
    return


app._unparsable_cell(
    r"""
    https://scikit-image.org/

    https://imageio.readthedocs.io/en/stable/


    https://scikit-image.org/docs/0.25.x/auto_examples/edges/plot_circular_elliptical_hough_transform.html
    """,
    name="_"
)


@app.cell
def _(plt):
    from skimage import data, color, img_as_ubyte
    from skimage.feature import canny
    from skimage.transform import hough_ellipse
    from skimage.draw import ellipse_perimeter

    # Load picture, convert to grayscale and detect edges
    image_rgb = data.coffee()[0:220, 160:420]
    image_gray = color.rgb2gray(image_rgb)
    edges = canny(image_gray, sigma=2.0, low_threshold=0.55, high_threshold=0.8)



    # Load picture, convert to grayscale and detect edges
    image_rgb = data.coffee()[0:220, 160:420]
    image_gray = color.rgb2gray(image_rgb)
    edges = canny(image_gray, sigma=2.0, low_threshold=0.55, high_threshold=0.8)

    # Perform a Hough Transform
    # The accuracy corresponds to the bin size of the histogram for minor axis lengths.
    # A higher `accuracy` value will lead to more ellipses being found, at the
    # cost of a lower precision on the minor axis length estimation.
    # A higher `threshold` will lead to less ellipses being found, filtering out those
    # with fewer edge points (as found above by the Canny detector) on their perimeter.
    result = hough_ellipse(edges, accuracy=20, threshold=250, min_size=100, max_size=120)
    result.sort(order='accumulator')

    # Estimated parameters for the ellipse
    best = list(result[-1])
    yc, xc, a, b = (int(round(x)) for x in best[1:5])
    orientation = best[5]

    # Draw the ellipse on the original image
    cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
    image_rgb[cy, cx] = (0, 0, 255)
    # Draw the edge (white) and the resulting ellipse (red)
    edges = color.gray2rgb(img_as_ubyte(edges))
    edges[cy, cx] = (250, 0, 0)

    fig2, (ax1, ax2) = plt.subplots(
        ncols=2, nrows=1, figsize=(8, 4), sharex=True, sharey=True
    )

    ax1.set_title('Original picture')
    ax1.imshow(image_rgb)

    ax2.set_title('Edge (white) and result (red)')
    ax2.imshow(edges)
    return canny, color


@app.cell
def _(plt):
    from skimage import io, filters, feature
    from skimage.color import rgb2gray
    from skimage.transform import hough_circle, hough_circle_peaks
    image_path = '/Users/valentingoupille/Desktop/Petri_analysis/image/test/1/Control4.JPG'
    image = io.imread(image_path)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    plt.show()
    gray_image = rgb2gray(image) if len(image.shape) == 3 else image
    edges_1 = feature.canny(gray_image, sigma=2, low_threshold=0.1, high_threshold=0.2)
    plt.imshow(edges_1, cmap='gray')
    plt.title('Edges detected')
    plt.axis('off')
    plt.show()
    return hough_circle, hough_circle_peaks, image, io, rgb2gray


@app.cell
def _(canny, color, image, io, plt):
    image_path_1 = '/Users/valentingoupille/Desktop/Petri_analysis/image/test/1/Control4.JPG'
    image_rgb_1 = io.imread(image_path_1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    plt.show()
    image_gray_1 = color.rgb2gray(image_rgb_1)
    plt.imshow(image_gray_1, cmap='gray')
    plt.title('Gray Image')
    plt.axis('off')
    plt.show()
    edges_2 = canny(image_gray_1, sigma=2.0, low_threshold=0.05, high_threshold=0.1)
    plt.imshow(edges_2, cmap='gray')
    plt.title('Edges detected')
    plt.axis('off')
    plt.show()
    return


@app.cell
def _(
    canny,
    color_1,
    hough_circle,
    hough_circle_peaks,
    image,
    io,
    np,
    plt,
    rgb2gray,
):
    from skimage import data, color
    from skimage.draw import circle_perimeter
    from skimage.util import img_as_ubyte
    image_path_2 = '/Users/valentingoupille/Desktop/Petri_analysis/image/test/1/Control4.JPG'
    image_rgb_2 = io.imread(image_path_2)
    plt.imshow(image_rgb_2)
    plt.title('Original Image')
    plt.axis('off')
    plt.show()
    image_rgb_2
    image_gray_2 = rgb2gray(image_rgb_2)
    edges_3 = canny(image_gray_2, sigma=1, low_threshold=1, high_threshold=2)
    plt.imshow(edges_3, cmap='gray')
    plt.title('Edges detected')
    plt.axis('off')
    plt.show()
    hough_radii = np.arange(100, 100, 2)
    hough_res = hough_circle(edges_3, hough_radii)
    accums, cx_1, cy_1, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
    image_1 = color_1.gray2rgb(image)
    for center_y, center_x, radius in zip(cy_1, cx_1, radii):
        circy, circx = circle_perimeter(center_y, center_x, radius, shape=image_1.shape)
        image_1[circy, circx] = (220, 20, 20)
    ax.imshow(image_1, cmap=plt.cm.gray)
    plt.show()
    return (color,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

