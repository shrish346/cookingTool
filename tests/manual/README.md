# Manual diagnostics — not pytest

These are standalone, **costed** scripts: they download real videos and call real
model APIs (need a populated `.env` and `ffmpeg`). pytest does not collect them
(`norecursedirs` in `pytest.ini`). Run from the repo root:

```bash
python3 tests/manual/test_pipeline.py [youtube_url]   # legacy frame-chunk path, end-to-end
python3 tests/manual/test_frames.py                   # FrameExtractor only (no API keys)
```

Note: `test_pipeline.py` exercises the legacy `analyze_scenes` frame-chunk path,
not the production `generate_recipe_direct` path. For a production-path live run,
see the pattern in the pytest suite's fakes (`tests/test_chef.py`) or drive
`RecipeChef.generate_recipe_direct` directly.
