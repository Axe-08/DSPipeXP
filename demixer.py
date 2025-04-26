import musdb
import numpy as np
from sklearn.decomposition import FastICA
import soundfile as sf
import itertools
import os

# Load MUSDB18 dataset (set 'root' to the path where MUSDB18 is stored)
mus = musdb.DB(root='path_to_musdb', subsets='train')

# Create an output directory for separated stems
os.makedirs('separated_stems', exist_ok=True)

# Iterate over tracks in the dataset
for track in mus.tracks:
    # Load the stereo mixture and sampling rate
    mixture = track.audio        # numpy array of shape (n_samples, 2)
    sr = track.rate              # Sampling rate (e.g., 44100 Hz)
    
    # Split into left and right channels
    mix_L = mixture[:, 0]
    mix_R = mixture[:, 1]
    
    # Create time-delayed copies of each channel to form multiple observations
    delay = 1024  # shift in samples (~0.02 sec at 44.1 kHz)
    mix_L_del = np.roll(mix_L, -delay)
    mix_R_del = np.roll(mix_R, -delay)
    
    # Stack the observations into a matrix for ICA
    # Each column is one “sensor” (observation): [Left, Right, Left_delay, Right_delay]
    X = np.vstack([mix_L, mix_R, mix_L_del, mix_R_del]).T  # shape (n_samples, 4)
    
    # Apply FastICA to estimate 4 independent source signals
    ica = FastICA(n_components=4, random_state=0)
    S = ica.fit_transform(X)   # Recovered signals, shape (n_samples, 4)
    
    # Each column of S is an estimated source (mono)
    sources = [S[:, i] for i in range(4)]
    
    # Retrieve ground-truth stems (vocals, drums, bass, other) as mono signals
    target_names = ['vocals', 'drums', 'bass', 'other']
    targets = [track.targets[name].audio for name in target_names]  # stereo arrays
    targets_mono = [np.mean(t, axis=1) for t in targets]  # convert to mono by averaging channels
    
    # Compute correlation between each estimated source and each target stem
    corr = np.zeros((4, 4))
    for i, src in enumerate(sources):
        for j, tgt in enumerate(targets_mono):
            # Avoid division by zero if silent
            if src.std() == 0 or tgt.std() == 0:
                corr[i, j] = 0
            else:
                corr[i, j] = np.corrcoef(src, tgt)[0, 1]
    
    # Find the best matching between estimated sources and stems by maximizing total correlation
    best_perm = None
    best_sum = -np.inf
    for perm in itertools.permutations(range(4)):
        total = sum(abs(corr[i, perm[i]]) for i in range(4))
        if total > best_sum:
            best_sum = total
            best_perm = perm
    
    # Map each target stem index to the index of the best-estimated source
    target_to_source = [None]*4
    for src_idx, tgt_idx in enumerate(best_perm):
        target_to_source[tgt_idx] = src_idx
    
    # Save each estimated source as a WAV file, labeled by the matched stem name
    safe_name = track.name.replace(' ', '_')  # sanitize track name for filenames
    for tgt_idx, stem_name in enumerate(target_names):
        src_idx = target_to_source[tgt_idx]
        estimated = sources[src_idx]
        # Duplicate mono signal to both stereo channels for output
        out_stereo = np.vstack([estimated, estimated]).T  # shape (n_samples, 2)
        out_path = f"separated_stems/{safe_name}_{stem_name}.wav"
        sf.write(out_path, out_stereo, sr)