"""Quick test for FrameExtractor"""
from src.downloaders.youtube import YouTubeDownloader
from src.processing.frames import FrameExtractor

# Download a short test video
print("Downloading test video...")
downloader = YouTubeDownloader()
video = downloader.download("https://www.youtube.com/watch?v=jNQXAC9IVRw")  # "Me at the zoo" - 19 seconds

print(f"  Title: {video.title}")
print(f"  Duration: {video.duration_seconds}")
print(f"  Path: {video.file_path}")

# Extract frames
print("\nExtracting frames...")
extractor = FrameExtractor(resize_width=640)
frames = extractor.extract(video.file_path)

print(f"  Extracted {len(frames)} frames")
print(f"  First frame base64 length: {len(frames[0])} chars")

# Cleanup
print("\nCleaning up...")
downloader.cleanup(video)
print("Done!")

