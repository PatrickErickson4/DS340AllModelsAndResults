# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt

## Original CLAHE file. We dont use it.

color = ['b', 'g', 'r']


def getColorHist(image):
    """
    This function produces channelwise histogram distribution.
    """
    if len(image.shape) == 3:
        hists = []
        for i, col in enumerate(color):
            hists.append(cv2.calcHist([image], [i], None, [256], [0, 256]))
        return hists
    else:
        hists = []
        i = 0
        hists.append(cv2.calcHist([image], [i], None, [256], [0, 256]))
        return hists


def applyCLAHE(image, display: bool = False):
    """
    CLAHE implementation.
    image: 3-channel RGB image from Keras.
    returns the image with CLAHE applied on luminance while preserving color.
    display: if True, it will display the input and output.
    """
    # 0) remember original dtype
    orig_dtype = image.dtype

    # 1) if float, convert to uint8 [0,255] but only once
    if image.dtype in (np.float32, np.float64):
        # if already in [0,1], scale up; otherwise assume [0,255]
        if image.max() <= 1.0:
            img_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        else:
            img_u8 = np.clip(image, 0, 255).astype(np.uint8)
    else:
        img_u8 = image  # already uint8

    # 2) convert RGB->LAB
    lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # 3) apply CLAHE on L channel (unchanged parameters)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(5,5))
    cl = clahe.apply(l)

    # 4) merge back and convert to RGB
    merged = cv2.merge([cl, a, b])
    clahe_img = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

    # 5) convert back to float32 [0,1] if needed
    if orig_dtype != np.uint8:
        clahe_img = clahe_img.astype(np.float32) / 255.0

    # optional display
    if display:
        fig = plt.figure(figsize=(9, 3), dpi=300)
        for idx, img in enumerate((img_u8, clahe_img)):
            ax = fig.add_subplot(1, 2, idx+1)
            ax.imshow(img if idx==1 else img_u8)
            ax.axis('off')
            ax.set_title(("Input","CLAHE")[idx])
        plt.show()

    return clahe_img


def applyHistogramEqualization(image, display: bool = False):
    """
    Applies the histogram equalization to a 3 channel grayscale image.
    If display is true, it wills show the comparison.
    """
    # https://123machinelearn.wordpress.com/2017/12/25/image-enhancement-using-high-frequency-emphasis-filtering-and-histogram-equalization/
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(image.shape)

    hist, bins = np.histogram(image_bw.flatten(), bins=256, range=[0, 256])
    cdf = hist.cumsum()
    cdf_norm = cdf * hist.max() / cdf.max()

    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    img_enhanced = cdf[image]

    if display:
        fig = plt.figure(figsize=(10, 5), dpi=300)
        rows, cols = 2, 2

        # Display the original image
        fig.add_subplot(rows, cols, 1)
        plt.imshow(image, cmap=plt.cm.gray);
        plt.axis('off');
        plt.title("Input Image")

        hists = getColorHist(image)
        fig.add_subplot(rows, cols, 3)
        plt.plot(hists[0], color="b")
        plt.plot(hists[1], color="g")
        plt.plot(hists[2], color="r")
        plt.xlim([0, 256])
        plt.title("Original Histogram")

        # Display the thresholded image
        fig.add_subplot(rows, cols, 2)
        plt.imshow(img_enhanced, cmap=plt.cm.gray);plt.axis('off')
        plt.title("Histogram Equalized Image")

        hists = getColorHist(img_enhanced)
        fig.add_subplot(rows, cols, 4)
        plt.plot(hists[0], color="b")
        plt.plot(hists[1], color="g")
        plt.plot(hists[2], color="r")
        plt.xlim([0, 256])
        plt.title("Equalized Histogram")

        plt.show()

        plt.plot

    print("final shape", img_enhanced.shape)
    return img_enhanced


def applyHFEFilter(image, display: bool = False):
    """
    This function applies the High Frequency Emphasis Filter on an Image.
    if display is true, it will show the comparison before and after the filter operation.
    """
    # https://123machinelearn.wordpress.com/2017/12/25/image-enhancement-using-high-frequency-emphasis-filtering-and-histogram-equalization/
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    npFFT = np.fft.fft2(image_bw)
    npFFTS = np.fft.fftshift(npFFT)

    # High-pass Gaussian filter
    (P, Q) = npFFTS.shape
    H = np.zeros((P, Q))
    D0 = 40
    for u in range(P):
        for v in range(Q):
            H[u, v] = 1.0 - np.exp(- ((u - P / 2.0) ** 2 + (v - Q / 2.0) ** 2) / (2 * (D0 ** 2)))
    k1 = 0.5;
    k2 = 0.80
    HFEfilt = k1 + k2 * H  # Apply High-frequency emphasis

    # Apply HFE filter to FFT of original image
    HFE = HFEfilt * npFFTS

    """
    Implement 2D-FFT algorithm

    Input : Input Image
    Output : 2D-FFT of input image
    """

    def fft2d(image):
        # 1) compute 1d-fft on columns
        fftcols = np.array([np.fft.fft(row) for row in image]).transpose()

        # 2) next, compute 1d-fft on in the opposite direction (for each row) on the resulting values
        return np.array([np.fft.fft(row) for row in fftcols]).transpose()

    # Perform IFFT (implemented here using the np.fft function)
    HFEfinal = (np.conjugate(fft2d(np.conjugate(HFE)))) / (P * Q)

    output = np.sqrt((HFEfinal.real) ** 2 + (HFEfinal.imag) ** 2)
    output = np.array(np.stack((output,) * 3, axis=-1), dtype=np.uint8)

    if display:
        fig = plt.figure(figsize=(10, 5), dpi=300)
        rows, cols = 1, 2

        # Display the original image
        fig.add_subplot(rows, cols, 1)
        plt.imshow(image, cmap=plt.cm.gray);
        plt.axis('off');
        plt.title("Input Image")

        # Display the thresholded image
        fig.add_subplot(rows, cols, 2)
        plt.imshow(output, cmap=plt.cm.gray);
        plt.axis('off')
        plt.title("HF Enhanced Image")
        plt.show()

    print(output.shape)
    return output


if __name__ == "__main__":
    img = cv2.imread("PlantVillage-Tomato/All-Tomato/Tomato___Bacterial_spot/image (1).JPG")

    applyCLAHE((img),display=True)
    #

    #applyHistogramEqualization(img, display=True)

    #applyHFEFilter(img, display=True)

    # applyHistogramEqualization(output, display=True)
