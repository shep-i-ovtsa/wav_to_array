import numpy as np
import scipy.io.wavfile as wav


def extract_frequencies(file_path):
    # Read the .wav file
    sample_rate, data = wav.read(file_path)

    # Check if stereo or mono
    if len(data.shape) == 2:
        # If stereo, separate channels
        left_channel = data[:, 0]
        right_channel = data[:, 1]
    else:
        left_channel = right_channel = data

    # Compute the Fast Fourier Transform (FFT) for both channels
    fft_left = np.fft.fft(left_channel)
    fft_right = np.fft.fft(right_channel)

    # Get frequencies
    freqs = np.fft.fftfreq(len(left_channel), d=1 / sample_rate)

    # Only keep the positive frequencies
    positive_freqs = freqs[freqs >= 0]

    # Magnitude of FFT for both channels
    magnitude_left = np.abs(fft_left)[:len(positive_freqs)]
    magnitude_right = np.abs(fft_right)[:len(positive_freqs)]

    return positive_freqs, magnitude_left, magnitude_right


# Example usage
file_path = 'C:\\Users\\shep\\Documents\\cyborg-ninja-kevin-macleod-main-version-7993-03-00.wav'
frequencies, left_magnitudes, right_magnitudes = extract_frequencies(file_path)

# Convert to usable format (e.g., picking the top N frequencies)
top_n = 5000  # You can adjust this number
left_notes = frequencies[np.argsort(left_magnitudes)[-top_n:]]
right_notes = frequencies[np.argsort(right_magnitudes)[-top_n:]]

# Format the output as comma-separated values
left_notes_str = ", ".join(f"{freq:.2f}" for freq in left_notes)
right_notes_str = ", ".join(f"{freq:.2f}" for freq in right_notes)

print("Left Channel Frequencies:", left_notes_str)
print("Right Channel Frequencies:", right_notes_str)
