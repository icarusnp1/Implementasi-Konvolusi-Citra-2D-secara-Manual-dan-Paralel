import cv2
import numpy as np
import time
from multiprocessing import Pool

def process_rows(args):
    start_row, end_row, image, kernel, H, W = args
    height, width = image.shape
    output = np.zeros((end_row - start_row, width), dtype=np.float32)

    for i in range(start_row, end_row):
        for j in range(W, width - W):
            total = 0.0
            for ki in range(-H, H+1):
                for kj in range(-W, W+1):
                    total += image[i + ki, j + kj] * kernel[H + ki, W + kj]
            output[i - start_row, j] = min(max(total, 0), 255)
    return (start_row, output)

def konvolusi_paralel(image, kernel, num_processes=4):
    height, width = image.shape
    H, W = kernel.shape[0] // 2, kernel.shape[1] // 2

    step = (height - 2 * H) // num_processes
    ranges = []
    for i in range(num_processes):
        start = H + i * step
        end = H + (i + 1) * step if i < num_processes - 1 else height - H
        ranges.append((start, end))

    args = [(start, end, image, kernel, H, W) for start, end in ranges]

    with Pool(processes=num_processes) as pool:
        results = pool.map(process_rows, args)

    out = np.zeros_like(image, dtype=np.float32)
    for start, data in results:
        out[start:start + data.shape[0], :] = data

    return out

# Pastikan semua kode ini ada dalam blok ini
if __name__ == '__main__':
    # Kernel 3x3 blur
    kernel = (1/9) * np.array([[1, 1, 1],
                               [1, 1, 1],
                               [1, 1, 1]], dtype=np.float32)

    # Baca gambar dan ubah ke grayscale
    Image = cv2.imread('image-3MP.JPG')
    if Image is None:
        print("Gambar tidak ditemukan.")
        exit()
    print("Gambar berhasil dibaca.")

    Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)

    # Padding gambar
    H, W = kernel.shape[0] // 2, kernel.shape[1] // 2
    Image_pad = np.pad(Image, ((H, H), (W, W)), mode='constant', constant_values=0)

    # Waktu mulai
    start_time = time.time()

    # Konvolusi paralel
    out = konvolusi_paralel(Image_pad, kernel, num_processes=4)

    # Waktu selesai
    elapsed = time.time() - start_time
    print(f"Waktu konvolusi (parallel blok baris): {elapsed:.4f} detik")

    # Simpan dan tampilkan hasil
    out = out.astype(np.uint8)
    
    cv2.imshow('Hasil Konvolusi Paralel', out)
    cv2.imwrite('output_parallel_safe.jpg', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
