import cv2
import numpy as np
import time
from numba import njit, prange

# Kernel 3x3 box blur
kernel = (1/9) * np.array(
    [[1, 1, 1],
     [1, 1, 1],
     [1, 1, 1]], dtype=np.float32)

@njit(parallel=True, fastmath=True)
def konvolusi_parallel(image, kernel):
    height, width = image.shape
    HK, WK = kernel.shape
    H = HK // 2
    W = WK // 2

    # Output array
    out = np.zeros_like(image, dtype=np.float32)

    for i in prange(H, height - H):  # Loop baris secara paralel
        for j in range(W, width - W):  # Kolom tetap serial untuk setiap baris
            sum = 0.0
            for k in range(-H, H + 1):
                for l in range(-W, W + 1):
                    a = image[i + k, j + l]
                    w = kernel[H + k, W + l]
                    sum += w * a
            out[i, j] = min(max(sum, 0), 255)  # clip secara manual
    return out

# --- MAIN PROGRAM ---
# Membaca gambar
Image = cv2.imread('DSC_0018.JPG')
if Image is None:
    print("Gambar tidak ditemukan.")
    exit()

Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)

# Padding agar bisa konvolusi di tepi
pad_h, pad_w = kernel.shape[0] // 2, kernel.shape[1] // 2
Image_pad = np.pad(Image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

# Proses konvolusi dengan waktu
start_time = time.time()
out = konvolusi_parallel(Image_pad, kernel)
elapsed = time.time() - start_time
print(f"Waktu konvolusi (parallel for): {elapsed:.4f} detik")

# Tampilkan dan simpan hasil
out = out.astype(np.uint8)
resized_out = cv2.resize(out, (800, 600))
cv2.imshow('Hasil Konvolusi Parallel', resized_out)
cv2.imwrite('output_parallel.jpg', out)
cv2.waitKey(0)
cv2.destroyAllWindows()
