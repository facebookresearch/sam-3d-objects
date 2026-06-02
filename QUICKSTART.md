# Quick Start Guide: SAM 3D Objects Web Application

This is a quick reference for getting the SAM 3D Objects unified web application up and running.

## What You Get

A single web application that provides:
1. **Object Tracker** - Track and segment objects in images/videos using text prompts
2. **Image Generator** - Generate images using kie.ai's Nano Banana service
3. **3D Creator** - Convert images to 3D models with interactive viewer

## Prerequisites

- Docker with GPU support (recommended), OR
- Python 3.12 + NVIDIA GPU with CUDA 12.6
- Node.js 18+ (for frontend development)
- Hugging Face account with SAM 3D Objects model access

## Option 1: Docker (Production)

```bash
# 1. Build frontend
cd frontend
npm install && npm run build
cd ..

# 2. Run with Docker
docker build -t sam3d-webapp -f backend/Dockerfile .
docker run -p 8000:8000 \
  -e HF_TOKEN="your_hf_token" \
  -e KIE_API_KEY="your_kie_key" \
  --gpus all \
  sam3d-webapp
```

Open http://localhost:8000

## Option 2: Development Mode

### Terminal 1 - Backend
```bash
# Install dependencies
pip install -r backend/requirements.txt
pip install -e .

# Set environment variables
export HF_TOKEN="your_hf_token"
export KIE_API_KEY="your_kie_key"

# Run server
python backend/server.py
```

### Terminal 2 - Frontend
```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `HF_TOKEN` | Hugging Face authentication token | Yes |
| `KIE_API_KEY` | kie.ai API key for image generation | For image gen |
| `KIE_API_ENDPOINT` | kie.ai API endpoint URL | No (has default) |
| `PORT` | Server port (default: 8000) | No |
| `HOST` | Server host (default: 0.0.0.0) | No |

## API Endpoints

Once running, the following endpoints are available:

- `GET /` - Web interface
- `GET /api/health` - Server health check
- `POST /api/process/image` - Process image with SAM 3D
- `POST /api/process/video` - Process video with SAM 3D
- `POST /api/generate/image-kie` - Generate image with kie.ai
- `POST /api/generate/3d` - Generate 3D model from image

## Troubleshooting

### "Module not found" errors
```bash
pip install -r backend/requirements.txt
pip install -e .
```

### Frontend won't build
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run build
```

### GPU/CUDA issues
- Verify NVIDIA drivers: `nvidia-smi`
- Check CUDA version: `nvcc --version`
- Ensure PyTorch sees GPU: `python -c "import torch; print(torch.cuda.is_available())"`

### Model loading fails
- Verify HF_TOKEN is set correctly
- Confirm access to SAM 3D Objects model on Hugging Face
- Check checkpoints are downloaded: `ls checkpoints/hf/`

## More Information

- Full setup guide: [WEBAPP_SETUP.md](WEBAPP_SETUP.md)
- Backend details: [backend/README.md](backend/README.md)
- Frontend details: [frontend/README.md](frontend/README.md)
- Original SAM 3D docs: [README.md](README.md)
