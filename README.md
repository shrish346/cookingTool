# MakerAI

<img width="1920" height="1030" alt="image" src="https://github.com/user-attachments/assets/831b3846-515f-4ec0-91ce-e1c584622d69" />
<img width="1914" height="1034" alt="image" src="https://github.com/user-attachments/assets/8bca1dbf-bfcf-4d15-b1d0-9651954ee957" />
<img width="1891" height="994" alt="image" src="https://github.com/user-attachments/assets/cdc4752d-c8e3-4b67-ae32-483bec2d6df3" />
<img width="1913" height="1026" alt="image" src="https://github.com/user-attachments/assets/6704a561-8286-4616-b95a-86f57db85a78" />

MakerAI is a tool that processes cooking youtube shorts, tik toks, and instagram reels into interactive, step-by-step recipes. It uses computer vision and AI to extract timestamps, generate instructions, and create looping video clips for each culinary step.

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

### Currently
Making this a pwa, so people can use it on their phones.
