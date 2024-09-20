import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load the audio file (generate a sine wave as a placeholder since actual audio file is not available)
sr = 22050  # Sample rate
t = np.linspace(0, 1.0, int(0.5 * sr))  # Time array for 0.5 second
y = 0.5 * np.sin(2 * np.pi * 440 * t)  # Generate a 440 Hz sine wave

# 1. Temporal View (Waveform)
plt.figure(figsize=(14, 5))
librosa.display.waveshow(y, sr=sr)
plt.title('Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Save the waveform plot
plt.savefig('waveform_plot.png')

# 2. Spectral View (Spectrogram) using librosa's stft function
plt.figure(figsize=(14, 5))
D = np.abs(librosa.stft(y))
DB = librosa.amplitude_to_db(D, ref=np.max)
librosa.display.specshow(DB, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')

# Save the spectrogram plot
plt.savefig('spectrogram_plot.png')
