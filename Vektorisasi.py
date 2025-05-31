import cv2
import numpy as np
import time

# Membuat kernel 3x3 box blur
kernel = np.ones((3, 3), dtype=np.float32) / 9  # atau pakai kernel kamu sendiri

# Membaca gambar dan konversi ke grayscale
Image = cv2.imread('image-24MP.jpg')
if Image is None:
    print("Gambar tidak ditemukan. Pastikan path gambar benar.")
    exit()

print("Gambar berhasil dibaca.")
Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)

# Padding gambar agar ukuran output tetap
H, W = kernel.shape
pad_h, pad_w = H // 2, W // 2
padded = np.pad(Image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

# Mulai hitung waktu
start_time = time.time()

# Membuat windows geser menggunakan striding
shape = (Image.shape[0], Image.shape[1], H, W)
strides = padded.strides * 2
windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)

# Konvolusi: element-wise multiply + sum
out = np.tensordot(windows, kernel, axes=((2, 3), (0, 1)))
out = np.clip(out, 0, 255).astype(np.uint8)

# Selesai hitung waktu
end_time = time.time()
elapsed = end_time - start_time
print(f"Waktu konvolusi (vektorisasi): {elapsed:.4f} detik")

# Tampilkan dan simpan hasil
resized_out = cv2.resize(out, (800, 600))  # agar tidak terlihat "zoom"
cv2.imshow('Hasil Konvolusi', resized_out)
cv2.imwrite('output_vektor.jpg', out)
cv2.waitKey(0)
cv2.destroyAllWindows()
