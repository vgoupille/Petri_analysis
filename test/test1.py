import marimo

__generated_with = "0.15.3"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt

    from skimage import data, color, img_as_ubyte
    from skimage.feature import canny
    from skimage.transform import hough_ellipse
    from skimage.draw import ellipse_perimeter
    return canny, color, img_as_ubyte, plt


@app.cell
def _():
    import glob
    import numpy as np
    return glob, np


@app.cell
def _(glob):
    from skimage.io import imread

    # import a set of images in a folder
    image_files = glob.glob("/Users/valentingoupille/Desktop/Petri_analysis/image/test/1/*.JPG")  # change extension as needed
    images = [imread(f) for f in image_files]

    images
    return (images,)


@app.cell
def _(images):
    len(images) # with f string
    return


@app.cell
def _(images):
    shapes = [image.shape for image in images]
    shapes
    return


@app.cell
def _(images, plt):
    # display the first image
    plt.imshow(images[0])    
    plt.show()
    return


@app.cell
def _(images, plt):

    n_images = len(images)
    n_rows = 2
    n_cols = (n_images + 1) // 2  # répartir sur 2 lignes

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    # axes peut être 1D si n_cols=1 → on le "flatten"
    axes = axes.ravel()

    for i, ax in enumerate(axes):
        if i < n_images:
            ax.imshow(images[i], cmap="gray")  # gris explicite
            ax.axis("off")
        else:
            ax.remove()  # supprime les cases inutilisées

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():

    # import seaborn as sns
    # import pandas as pd

    # # Créer un DataFrame pour que seaborn gère l'organisation
    # df = pd.DataFrame({"image": images})

    # # Ajouter un id pour positionner dans la grille
    # df["id"] = range(len(images))

    # # Déterminer la grille : 2 lignes
    # n_rows = 2
    # n_cols = (len(images) + 1) // 2

    # # Construire la grille
    # g = sns.FacetGrid(df, col="id", col_wrap=n_cols, height=4)

    # # Mapper chaque image dans la grille
    # for ax, img in zip(g.axes.flat, images):
    #     ax.imshow(img, cmap="gray")
    #     ax.axis("off")

    # plt.tight_layout()
    # plt.show()
    return


@app.cell
def _(color, images, img_as_ubyte):
    # compresed version of the images 
    compressed_images = [img_as_ubyte(color.rgb2gray(img)) if len(img.shape) == 3 else img_as_ubyte(img) for img in images]
    compressed_images
    print([img.shape for img in compressed_images])
    return (compressed_images,)


@app.cell
def _(compressed_images, plt):
    # view the compressed images
    # display the first image
    plt.imshow(compressed_images[0])    
    plt.show()
    return


@app.cell
def _(compressed_images, images):
    # compare the original and compressed images
    for d, img in enumerate(images):
        print(f"\nImage {d} AVANT :")
        print(f"  shape = {img.shape}, dtype = {img.dtype}, min = {img.min()}, max = {img.max()}")

        compressed = compressed_images[d]
        print(f"Image {d} APRÈS :")
        print(f"  shape = {compressed.shape}, dtype = {compressed.dtype}, min = {compressed.min()}, max = {compressed.max()}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        1️⃣ Avant conversion
        	•	Toutes les images sont couleur (4000, 6000, 3).
        	•	dtype = uint8, valeurs entre 0 et 255.
        	•	Donc chaque pixel est codé sur 3 canaux (R, G, B).
    
        ⸻
    
        2️⃣ Après conversion (rgb2gray + img_as_ubyte)
        	•	Toutes les images deviennent grayscale (4000, 6000), 2D.
        	•	Toujours dtype = uint8.
        	•	Min n’est plus 0 mais ~8–13
        	•	C’est normal : rgb2gray fait un mélange pondéré des trois canaux.
        	•	Par exemple, Y = 0.2126 R + 0.7152 G + 0.0722 B.
        	•	Les pixels très sombres ne deviennent donc plus exactement 0 après conversion et passage à uint8.
        	•	Max reste 255 → les pixels clairs restent clairs.
    
        ⸻
    
        3️⃣ Points importants à retenir
        	•	La conversion perd l’information couleur mais conserve les contrastes.
        	•	Les valeurs minimales supérieures à 0 ne posent généralement pas de problème pour des traitements comme Canny, seuillage ou détection de contours.
        """
    )
    return


@app.cell
def _(compressed_images, np):
    def normalize_uint8(img):
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min()) * 255
        return img.astype(np.uint8)

    normalized_images = [normalize_uint8(img) for img in compressed_images]
    return (normalized_images,)


@app.cell
def _(compressed_images, normalized_images):
    # compare the compressed and normalized images
    for b, imgc in enumerate(compressed_images):
        print(f"\nImage {b} AVANT normalisation :")
        print(f"  shape = {imgc.shape}, dtype = {imgc.dtype}, min = {imgc.min()}, max = {imgc.max()}")

        norm = normalized_images[b]
        print(f"Image {b} APRÈS :")
        print(f"  shape = {norm.shape}, dtype = {norm.dtype}, min = {norm.min()}, max = {norm.max()}")
    return


@app.cell
def _(compressed_images, normalized_images, plt):
    # view the compressed images
    # display the first image
    plt.imshow(compressed_images[0])    
    plt.show()

    # display the normalized image
    plt.imshow(normalized_images[0])
    plt.show()
    return


@app.cell
def _(compressed_images, normalized_images, plt):
    plt.imshow(compressed_images[0], cmap="gray")  # préciser le cmap
    #plt.axis("off")  # optionnel, pour enlever les axes
    plt.show()


    plt.imshow(normalized_images[0], cmap="gray")  # préciser le cmap
    #plt.axis("off")  # optionnel, pour enlever les axes
    plt.show()
    return


@app.cell
def _(canny, normalized_images, plt):
    # edge detection with canny
    edges = canny(normalized_images[0], sigma=5, low_threshold=10, high_threshold=50)
    plt.imshow(edges, cmap='gray')
    plt.show()
    return (edges,)


@app.cell
def _(edges, np):
    from skimage.transform import hough_circle, hough_circle_peaks
    # Detect two radii 
    hough_radii = np.arange(1500, 3000, 500)
    hough_res = hough_circle(edges, hough_radii)
    return hough_circle_peaks, hough_radii, hough_res


@app.cell
def _(hough_circle_peaks, hough_radii, hough_res):
    # Select the most prominent 1 circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)
    return accums, cx, cy, radii


@app.cell
def _(hough_res):
    hough_res
    return


@app.cell
def _(accums, cx, cy, radii):
    print("Votes des cercles :", accums)
    print("Centres X :", cx)
    print("Centres Y :", cy)
    print("Rayons :", radii)
    return


@app.cell
def _(color, cx, cy, normalized_images, plt, radii):
    from skimage.draw import circle_perimeter



    # Draw them
    figu, axez = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))

    image0 = normalized_images[0]
    image0 = color.gray2rgb(image0)
    for center_y, center_x, radius in zip(cy, cx, radii):
        circy, circx = circle_perimeter(center_y, center_x, radius, shape=image0.shape)
        image0[circy, circx] = (220, 20, 20)

    axez.imshow(image0, cmap=plt.cm.gray)
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"ensuite, on va vouloir detecter la forme de cercle dans les images. Pour cela, on va utiliser la transformée de Hough.")
    return


if __name__ == "__main__":
    app.run()
