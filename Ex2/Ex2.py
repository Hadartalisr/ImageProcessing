import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import read, write
import matplotlib as mpl

mpl.use('macosx')

def read_raw_y(file_path):
    if not os.path.isfile(file_path):
        raise ValueError("Input file is wrong")

    (fs, y) = read(file_path)
    return fs, y


def write_wav_file(file_path, fs, y):
    write(file_path, fs, y)


(fs, signal) = read_raw_y('./assets/q1.wav')
signal_len = len(signal)
t = np.linspace(0, signal_len / fs, signal_len, endpoint=False)  # time

dft_result = np.fft.fft(signal)
frequencies = np.fft.fftfreq(signal_len, 1 / fs)




# Read the WAV file
filename = './assets/q1.wav'
fs, audio_data = read(filename)

# Time vector
t = np.arange(len(audio_data)) / fs

# Perform Discrete Fourier Transform (DFT)
dft_result = np.fft.fft(audio_data)
frequencies = np.fft.fftfreq(len(audio_data), 1 / fs)

# Set high frequencies to zero (filtering)
cutoff_frequency = 1700  # Adjust the cutoff frequency as needed
dft_result_filtered = dft_result.copy()
dft_result_filtered[np.abs(frequencies) > cutoff_frequency] = 0

# Perform Inverse Discrete Fourier Transform (IDFT)
audio_data_filtered = np.fft.ifft(dft_result_filtered)

# Plot the original and filtered signals
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.plot(t, audio_data)
plt.title('Original Audio Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(2, 2, 2)
plt.plot(frequencies, dft_result)
plt.title('Original Audio Signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')

plt.subplot(2, 2, 3)
plt.plot(t, np.real(audio_data_filtered))
plt.title('Filtered Audio Signal (High Frequencies Removed)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()

write('./assets/q1-filtered.wav', fs, np.real(audio_data_filtered))
max_index = 1000
plt.plot(frequencies[max_index - 100:max_index + 100], dft_result[max_index - 100:max_index + 100])
max_freq = dft_result[max_index]
