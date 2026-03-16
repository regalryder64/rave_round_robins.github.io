import os, librosa, soundfile as sf
import numpy as np

dataset_path = 'mediumsnare_wav'
for f in os.listdir(dataset_path):
    if f.endswith('.wav'):
        path = os.path.join(dataset_path, f)
        y, sr = librosa.load(path, sr=44100)

        # Pad to 1.5 seconds
        y_padded = librosa.util.fix_length(y, size=int(sr * 1.5))

        # ADD DITHER: Add microscopic noise to prevent dividing by zero in Phase 2
        #noise = np.random.randn(*y_padded.shape) * 1e-6
        #y_padded = y_padded + noise

        sf.write(path, y_padded, sr)