# Chef's Loop

image here

Chef's Loop is a tool that processes cooking youtube shorts, tik toks, and instagram reels into interactive, step-by-step recipes. It uses computer vision and AI to extract timestamps, generate instructions, and create looping video clips for each culinary step.

## Docker Setup

1. Copy `env.example` to `.env` and fill in your API keys.
2. Start the services:

```bash
docker-compose up --build
```

The frontend is available at `http://localhost:5173` and the backend at `http://localhost:8000`.

## Local Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- FFmpeg
- Redis

### Backend

1. Copy `env.example` to `.env` and fill in your API keys.
2. Install dependencies:

```bash
pip install -r backend/requirements.txt
```

3. Start the server from the project root:

```bash
uvicorn backend.api.main:app --reload
```

### Frontend

1. Navigate to the frontend directory:

```bash
cd frontend
```

2. Install dependencies and start the development server:

```bash
npm install
npm run dev
```
