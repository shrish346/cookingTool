# MakerAI

MakerAI turns cooking videos — YouTube Shorts, TikToks, Instagram Reels — into interactive step-by-step recipes you can actually cook from. Every step is backed by a looping video clip cut from the original video, so instead of scrubbing back and forth through a 45-second short, you see exactly the moment that matters for the step you're on.

**Try it:** [makeraibeta.vercel.app](https://makeraibeta.vercel.app/)

**Demo:** [Watch it here](https://drive.google.com/file/d/1axwNX-d5Gw55TFWKvaDuWC6ToaCGVNGu/view?usp=sharing)

<img width="1920" height="1030" alt="image" src="https://github.com/user-attachments/assets/831b3846-515f-4ec0-91ce-e1c584622d69" />
<img width="1914" height="1034" alt="image" src="https://github.com/user-attachments/assets/8bca1dbf-bfcf-4d15-b1d0-9651954ee957" />
<img width="1891" height="994" alt="image" src="https://github.com/user-attachments/assets/cdc4752d-c8e3-4b67-ae32-483bec2d6df3" />
<img width="1913" height="1026" alt="image" src="https://github.com/user-attachments/assets/6704a561-8286-4616-b95a-86f57db85a78" />

## The problem

Short-form cooking videos are the most popular recipe format on the internet, and the worst one to cook from. They're fast, they skip prep, they rarely state quantities, and the one detail you need is buried somewhere in a 30-second clip you have to rewatch five times with wet hands.

## What you get

Paste a link, and a few minutes later you have:

- A step-by-step recipe where each step loops the exact seconds of video it came from
- Real quantities, even when the video never stated them
- A doneness cue on every step, so you know what "done" looks like
- Steps the video skipped — the prep it glossed over, the technique it assumed you knew
- A full ingredient and tool list, with two setup steps that walk you through gathering everything
- A cooking mode built for the kitchen: large text, tap anywhere to advance, and the screen never dims mid-step

Every ingredient and step is labeled with where it came from: seen in the video, pulled from a published recipe (with the source cited), or estimated by the model. You always know what the camera showed versus what was filled in.

## How recipe generation works

The hard problem is temporal grounding: mapping each written step back to the exact seconds of video it came from, without the AI making timestamps up. The pipeline is built so it can't.

1. **Watch.** A vision model watches the full video and produces a scene log: every atomic cooking action ("add salt", "flip the chicken") with the timestamp it happened at.

2. **Ground.** A language model turns that log into a recipe skeleton. It is only allowed to report what the camera showed, and each step must cite the specific logged actions it was derived from. It never writes timestamps — it cites action IDs, and code deterministically resolves those into start and end times. Timing is computed, not generated.

3. **Expand.** A second model pass, with live web access, rewrites the skeleton for someone who has never cooked before. It pulls real quantities from published recipes, adds a doneness cue to every step, and inserts the steps the video skipped. It is never shown a timestamp and never writes one — the grounded timing is frozen and re-attached in code after it runs, so the expansion can add knowledge but cannot drift the video alignment.

4. **Cut.** FFmpeg cuts a clip for each grounded step from the source video. Steps the expansion invented have no clip — because there's no footage of them — and the recipe says so rather than faking it.

The recipe text appears as soon as it's ready; clips stream in behind it. Processed videos are cached, so a video anyone has already run comes back instantly.

## Platforms

YouTube Shorts, TikTok, and Instagram Reels. MakerAI is a progressive web app — install it to your phone's home screen and it works like a native app in the kitchen.
