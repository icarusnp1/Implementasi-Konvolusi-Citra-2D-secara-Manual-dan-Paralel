import cv2
import numpy as np
import time

# Membuat kernel 11x11 box blur
# kernel = np.ones((11, 11), dtype=np.float32) / 121
kernel = (1/9) * np.array(
                            [[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]])

# Membaca gambar dan konversi ke grayscale
Image = cv2.imread('image-24MP.JPG')
if Image is None:
    print("Gambar tidak ditemukan. Pastikan path gambar benar.")
else:
    print("Gambar berhasil dibaca.")
    
Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)

height, width = Image.shape
HK, WK = kernel.shape

# Menentukan padding (jarak tepi)
H = HK // 2
W = WK // 2

# Inisialisasi output image
out = np.zeros_like(Image, dtype=np.float32)

# Mulai hitung waktu
start_time = time.time()

# Operasi konvolusi manual
for i in range(H, height - H):
    for j in range(W, width - W):
        sum = 0.0
        for k in range(-H, H + 1):
            for l in range(-W, W + 1):
                a = Image[i + k, j + l]
                w = kernel[H + k, W + l]
                sum += w * a
        out[i, j] = np.clip(sum, 0, 255)

# Selesai hitung waktu
end_time = time.time()
elapsed = end_time - start_time
print(f"Waktu konvolusi: {elapsed:.4f} detik")

# Konversi output ke uint8 untuk ditampilkan
out = out.astype(np.uint8)

# Tampilkan hasil
cv2.imshow('Hasil Konvolusi', out)
cv2.imwrite('output.jpg', out)
cv2.waitKey(0)
cv2.destroyAllWindows()
