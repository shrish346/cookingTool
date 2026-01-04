import sys
import os
import time
import base64
import subprocess
import tempfile
import shutil
from dotenv import load_dotenv
from openai import OpenAI
import yt_dlp

load_dotenv()

# Configuration
DEFAULT_TEST_URL = "https://www.youtube.com/shorts/Ll307dapW64"
MODEL_ID = "google/gemini-2.0-flash-001"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def download_video(url: str) -> tuple[str, str, int]:
    temp_dir = tempfile.mkdtemp()
    ydl_opts = {
        'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
        'format': 'best[height<=480][ext=mp4]/best[height<=480]', # Get small version
        'quiet': True,
        'no_warnings': True,
    }
    
    print(f"  Downloading from {url}...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get('title', 'Unknown')
        duration = int(info.get('duration', 0))
        files = [f for f in os.listdir(temp_dir) if os.path.isfile(os.path.join(temp_dir, f))]
        return os.path.join(temp_dir, files[0]), title, duration

def compress_video_hard(input_path: str) -> str:
    """
    Aggressively compresses video to ensure it fits in API payload (<10MB).
    API 'native' video support usually fails if the base64 string is too massive.
    """
    fd, output_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    
    print(f"  Compressing video to fit API limits...")
    
    # Resize to 360p, reduce frame rate to 2fps (enough for VLM), low bitrate
    # This keeps the file physically small while keeping it a valid "video" container
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vf", "scale=-2:360,fps=2", 
        "-c:v", "libx264", "-crf", "32", "-preset", "ultrafast",
        "-an", # Remove audio to save space
        output_path
    ]
    
    subprocess.run(cmd, check=True, capture_output=True)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Final Size: {size_mb:.2f} MB")
    
    if size_mb > 15:
        print("  WARNING: Video is >15MB. API might reject it.")
        
    return output_path

def main(test_url: str = DEFAULT_TEST_URL):
    if not OPENROUTER_API_KEY:
        print("Error: OPENROUTER_API_KEY not found.")
        return

    print(f"  VLM NATIVE VIDEO TEST")
    print("=" * 60)

    try:
        # 1. Download
        raw_path, title, duration = download_video(test_url)
        
        # 2. Compress (Required for Base64 transfer)
        video_path = compress_video_hard(raw_path)
        
        # 3. Encode
        with open(video_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")

        # 4. Send to API
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )

        print(f"\n  Sending to {MODEL_ID}...")
        
        # CORRECT SCHEMA for OpenRouter Video
        prompt = f"""You are a forensic video analyst specializing in culinary processes. Your goal is to create a factual log of cooking actions based ONLY on visual evidence.

*** STEP 1: VISUAL SCRATCHPAD (Mandatory) ***
Watch the video chronologically. Create a raw text log of distinct physical movements.
- Format: [Start MM:SS - End MM:SS] : [Entity] -> [Action]
- Multitasking: If multiple actions occur simultaneously (e.g., stirring while pouring), list them as SEPARATE lines with the same timestamp range.
- Constraint: If nothing significant happens for 5 seconds, do NOT write anything.

*** STEP 2: JSON GENERATION ***
Convert your scratchpad into valid JSON.

CRITICAL FORMATTING RULES:
1. Do not nest actions inside other actions. Every action is a separate object in the main list.
2. Check your brackets. Ensure every '{{' has a closing '}}' before starting a new item.
3. No Duplicates: Do not repeat the same action for the same timestamp.

JSON EXAMPLE (Follow this structure exactly):
{{
  "summary": "...",
  "entities": [ ... ],
  "micro_actions": [
    {{
      "action": "Slicing Cucumber into Rounds",
      "start": "00:28",
      "end": "00:30",
      "entity": "Knife",
      "concurrent_with_other_action": false
    }},
    {{
       ... next action ...
    }}
  ]
}}

STRICT ACCURACY RULES:
1. Visual Evidence Only: If you do not see a hand touching an object or a tool moving, do not list it.
2. No Filler: Quality over quantity. Do not hallucinate generic actions just to reach a higher count.
3. Valid Actions: Include both STATE CHANGES (e.g., chopping, melting) and ACTIVE PROCESSES (e.g., stirring, whisking, basting). Ignore passive states (e.g., "soup is boiling" with no human interaction).
   - Output format for valid actions: [Action Verb] + [Object] + [Resulting State/Location]
   - Transformation: If the object changes shape/form, describe the new form (e.g., "slicing carrots into rounds").
   - Destination: If the object moves, describe the container (e.g., "transferring onions to the hot skillet").
   - Completion: If the action is a process, describe the goal state (e.g., "whisking eggs until frothy").
   - Accuracy: If the resulting state/location is not clear, do not list it.
4. Concurrency: You MUST list simultaneous actions separately. Do not group them into one generic event.
5. Entity Consistency: Use the exact same name for an entity in the 'micro_actions' list as you defined in the 'entities' list.
6. Time Format: Output timestamps as "MM:SS" strings in the JSON to avoid math errors.

Return ONLY the Scratchpad followed by the JSON object."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": prompt
                    },
                    {
                        "type": "video_url",  # <--- This must be 'video_url', NOT 'video'
                        "video_url": {
                            "url": f"data:video/mp4;base64,{video_b64}"
                        }
                    }
                ]
            }
        ]

        start = time.perf_counter()
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
        )
        print(f"  Time: {time.perf_counter() - start:.2f}s")
        print("-" * 60)
        print(response.choices[0].message.content)
        print("-" * 60)

    except Exception as e:
        print(f"\nErro1r: {e}")
        # Helpful debugging for "400 Bad Request"
        if "400" in str(e):
            print("\nNOTE: A 400 error usually means the video file is still too big for the API.")

    finally:
        # Cleanup
        try:
            if 'raw_path' in locals() and os.path.dirname(raw_path).startswith(tempfile.gettempdir()):
                shutil.rmtree(os.path.dirname(raw_path))
            if 'video_path' in locals() and os.path.exists(video_path):
                os.remove(video_path)
        except:
            pass

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEST_URL
    main(url)