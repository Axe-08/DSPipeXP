import kagglehub

# Download latest version
path = kagglehub.dataset_download("imuhammad/audio-features-and-lyrics-of-spotify-songs")

print("Path to dataset files:", path)

#then mv ${copied file path} ~/DSPipeXP/lyric_dataset/
