import cv2
import numpy as np
import time
from numba import njit, prange
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np

def read_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Gambar tidak ditemukan: {image_path}")
        return None
    print(f"Gambar berhasil dibaca: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # Convert to grayscale to simplify processing

def vanilla(img, kernel):
    height, width = img.shape
    HK, WK = kernel.shape

    # Menentukan padding (jarak tepi)
    H = HK // 2
    W = WK // 2

    # Inisialisasi output image
    out = np.zeros_like(img, dtype=np.float32)

    # Operasi konvolusi manual
    for i in range(H, height - H):
        for j in range(W, width - W):
            sum = 0.0
            for k in range(-H, H + 1):
                for l in range(-W, W + 1):
                    a = img[i + k, j + l]
                    w = kernel[H + k, W + l]
                    sum += w * a
            out[i, j] = np.clip(sum, 0, 255)

def vektorisasi(img, kernel):
    H, W = kernel.shape
    pad_h, pad_w = H // 2, W // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    
    # Membuat windows geser menggunakan striding
    shape = (img.shape[0], img.shape[1], H, W)
    strides = padded.strides * 2
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)

    # Konvolusi: element-wise multiply + sum
    out = np.tensordot(windows, kernel, axes=((2, 3), (0, 1)))
    out = np.clip(out, 0, 255).astype(np.uint8)

@njit(parallel=True, fastmath=True)
def parallel_loop(img, kernel):
    height, width = img.shape
    HK, WK = kernel.shape
    H = HK // 2
    W = WK // 2

    # Output array
    out = np.zeros_like(img, dtype=np.float32)

    for i in prange(H, height - H):  # Loop baris secara paralel
        for j in range(W, width - W):  # Kolom tetap serial untuk setiap baris
            sum = 0.0
            for k in range(-H, H + 1):
                for l in range(-W, W + 1):
                    a = img[i + k, j + l]
                    w = kernel[H + k, W + l]
                    sum += w * a
            out[i, j] = min(max(sum, 0), 255)  # clip secara manual
    return out

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

def multiprocess(img, kernel, num_processes=4):
    height, width = img.shape
    H, W = kernel.shape[0] // 2, kernel.shape[1] // 2

    step = (height - 2 * H) // num_processes
    ranges = []
    for i in range(num_processes):
        start = H + i * step
        end = H + (i + 1) * step if i < num_processes - 1 else height - H
        ranges.append((start, end))

    args = [(start, end, img, kernel, H, W) for start, end in ranges]

    with Pool(processes=num_processes) as pool:
        results = pool.map(process_rows, args)

    out = np.zeros_like(img, dtype=np.float32)
    for start, data in results:
        out[start:start + data.shape[0], :] = data

    return out

if __name__ == "__main__":
    # Here is a test to benchmark different methods of computing the same task.
    kernel = (1/9) * np.array([[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]])

    # Image incrementing from 3 MP to 24 MP
    img1 = read_image('image-3MP.jpg')
    img2 = read_image('image-9MP.jpg')
    img3 = read_image('image-15MP.jpg')
    img4 = read_image('image-21MP.jpg')
    img5 = read_image('image-24MP.jpg')

    list_img = [img1, img2, img3, img4, img5]
    
    # Padding agar bisa konvolusi di tepi
    pad_h, pad_w = kernel.shape[0] // 2, kernel.shape[1] // 2
    Image_pad1 = np.pad(img1, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    
    # Warm up Numba JIT before timing
    parallel_loop(Image_pad1, kernel)  # Call once, ignore result

    megapixels = [3, 9, 15, 21, 24]
    vanilla_times = []
    vektorisasi_times = []
    parallel_times = []
    multiprocess_times = []

    for i, img in enumerate(list_img):
        if img is None:
            continue
        print(f"Testing image {i+1} with size {img.shape}")
        
        # Padding agar bisa konvolusi di tepi
        pad_h, pad_w = kernel.shape[0] // 2, kernel.shape[1] // 2
        Image_pad = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
        
        # ==================== Vanilla method ==================== 
        start_time = time.time()
        vanilla(img, kernel)
        elapsed = time.time() - start_time
        vanilla_times.append(elapsed)
        print(f"Vanilla method took {elapsed:.4f} seconds")
        
        # ==================== Vektorisasi method ==================== 
        start_time = time.time()
        vektorisasi(img, kernel)
        elapsed = time.time() - start_time
        vektorisasi_times.append(elapsed)
        print(f"Vektorisasi method took {elapsed:.4f} seconds")
        
        # ==================== Parallel loop method ====================     
        start_time = time.time()
        parallel_loop(Image_pad, kernel)
        elapsed = time.time() - start_time
        parallel_times.append(elapsed)
        print(f"Parallel loop method took {elapsed:.4f} seconds")
        
        # ==================== Multiprocess method ==================== 
        start_time = time.time()
        multiprocess(Image_pad, kernel, num_processes=4)
        elapsed = time.time() - start_time
        multiprocess_times.append(elapsed)
        print(f"Multiprocess method took {elapsed:.4f} seconds")
        
        
        
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(megapixels[:len(vanilla_times)], vanilla_times, marker='o', label='Vanilla')
    plt.plot(megapixels[:len(vektorisasi_times)], vektorisasi_times, marker='o', label='Vektorisasi')
    plt.plot(megapixels[:len(parallel_times)], parallel_times, marker='o', label='Parallel Loop')
    plt.plot(megapixels[:len(multiprocess_times)], multiprocess_times, marker='o', label='Multiprocess')

    plt.yscale('log')
    plt.yticks([1e-1, 1, 10, 100, 1000, 10000], ['10$^{-1}$', '10$^{0}$', '10$^{1}$', '10$^{2}$', '10$^{3}$', '10$^{4}$'])
    plt.xlabel('Image Size (Mega Pixels)')
    plt.ylabel('Elapsed Time (seconds, log scale)')
    plt.title('Benchmarking Convolution Methods')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()