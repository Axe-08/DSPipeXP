#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from music_rec_system import MusicRecommender
import yt_dlp

class YouTubeDownloader:
    """
    A simple YouTube downloader using yt-dlp to fetch the best audio stream.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def search_and_download(self, query: str) -> str | None:
        # Use yt-dlp's "ytsearch1:" prefix to auto-search and download the first result
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(self.output_dir, '%(title)s.%(ext)s'),
            'noplaylist': True,
            'quiet': True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"ytsearch1:{query}", download=True)
                # info['entries'][0] contains metadata for the downloaded item
                if info and 'entries' in info and len(info['entries']) > 0:
                    entry = info['entries'][0]
                    # Construct the filename yt-dlp used
                    filename = ydl.prepare_filename(entry)
                    return filename
        except Exception as e:
            print(f"Error downloading '{query}': {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Music Recommendation System')
    parser.add_argument('--song', type=str, help='Path to a song file or name of a song to search')
    parser.add_argument('--method', type=str, choices=['knn', 'cosine', 'hybrid'], \
                        default='hybrid', help='Recommendation method to use')
    parser.add_argument('--count', type=int, default=5, \
                        help='Number of recommendations to return')
    parser.add_argument('--search', type=str, help='Search for a song by name or artist')
    parser.add_argument('--download', action='store_true', \
                        help='Download song from YouTube if not found locally')
    parser.add_argument('--no-download', dest='download', action='store_false',
                        help='Don\'t download songs from YouTube')
    parser.set_defaults(download=True)
    args = parser.parse_args()

    # Initialize the recommender
    base_dir = Path(__file__).parent
    dataset_path = base_dir / 'spotify_dataset' / 'dataset.csv'
    recommender = MusicRecommender(str(dataset_path))

    def handle_results(results, label):
        if results.empty:
            print(f"No {label} found in the dataset.")
        else:
            print(f"Found {len(results)} matching songs:")
            print(results)

    # SEARCH FLOW
    if args.search:
        print(f"Searching for: {args.search}")
        results = recommender.search_song(args.search)
        if results.empty and args.download:
            print("Not found locally, attempting YouTube download...")
            downloader = YouTubeDownloader(output_dir=str(base_dir))
            downloaded = downloader.search_and_download(args.search)
            if downloaded:
                print(f"Downloaded: {os.path.basename(downloaded)}")
                recommender.add_song_to_dataset(downloaded)
                recs = recommender.recommend_songs(downloaded, n_recommendations=args.count, method=args.method)
                print(f"\nTop {args.count} recommendations ({args.method}):")
                print(recs)
            else:
                print("Download failed.")
        else:
            handle_results(results, 'matches')
        return

    # SONG SPECIFIED FLOW
    if not args.song:
        parser.print_help()
        return

    song_input = args.song
    song_path = Path(song_input)

    # If not a file, search dataset
    if not song_path.is_file():
        print(f"Searching for song: {song_input}")
        results = recommender.search_song(song_input)
        if results.empty:
            if args.download:
                print("Not found, downloading from YouTube...")
                downloader = YouTubeDownloader(output_dir=str(base_dir))
                downloaded = downloader.search_and_download(song_input)
                if downloaded:
                    print(f"Downloaded: {os.path.basename(downloaded)}")
                    song_path = Path(downloaded)
                    recommender.add_song_to_dataset(str(song_path))
                else:
                    print("Download failed. Please provide a valid .mp3 path.")
                    return
            else:
                print("Song not found locally and download disabled.")
                return
        else:
            handle_results(results, 'matches')
            idx = input("\nEnter the number of the song to use (1-based), or 0 to cancel: ")
            try:
                choice = int(idx) - 1
                if choice < 0 or choice >= len(results): raise ValueError
                row = results.iloc[choice]
                song_path = None
                # Use the dataset features directly
                recs = recommender.recommend_from_index(
                    index=results.index[choice],
                    n_recommendations=args.count,
                    method=args.method
                )
                print(f"\nTop {args.count} recommendations ({args.method}):")
                print(recs)
            except ValueError:
                print("Cancelled or invalid choice.")
            return

    # At this point song_path is a valid file
    song_file = str(song_path)
    print(f"Analyzing song file: {song_file}")
    recommender.add_song_to_dataset(song_file)
    recs = recommender.recommend_songs(song_file, n_recommendations=args.count, method=args.method)
    print(f"\nTop {args.count} recommendations ({args.method}):")
    print(recs)


if __name__ == "__main__":
    main()
