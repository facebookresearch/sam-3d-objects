# SAM 3D Objects Backend

FastAPI server for SAM 3D Objects unified web application. This server serves both the AI API endpoints and the React frontend from a single process.

## Features

- **Unified Serving**: Serves both API endpoints and React frontend
- **SAM 3D Integration**: Complete integration with SAM 3D Objects model
- **Multiple Endpoints**:
  - `/api/process/image` - Process images with SAM 3D
  - `/api/process/video` - Process videos with SAM 3D (video predictor)
  - `/api/generate/image-kie` - Generate images using kie.ai's Nano Banana service
  - `/api/generate/3d` - Generate 3D models from images

## Requirements

- Python 3.12
- CUDA 12.6 (for GPU acceleration)
- PyTorch 2.7 with CUDA 12.6 support

## Installation

### Using Docker (Recommended)

```bash
# Build the Docker image
docker build -t sam3d-backend -f backend/Dockerfile .

# Run the container
docker run -p 8000:8000 \
  -e HF_TOKEN="your_huggingface_token" \
  -e KIE_API_KEY="your_kie_api_key" \
  --gpus all \
  sam3d-backend
```

### Local Development

1. **Install dependencies:**

```bash
pip install -r backend/requirements.txt
```

2. **Clone and install SAM 3:**

```bash
git clone https://github.com/facebookresearch/sam3.git /tmp/sam3
cd /tmp/sam3
pip install -e .
```

3. **Install SAM 3D Objects package:**

```bash
pip install -e .
```

4. **Set environment variables:**

```bash
export HF_TOKEN="your_huggingface_token"
export KIE_API_KEY="your_kie_api_key"
```

5. **Run the server:**

```bash
python backend/server.py
```

The server will start on `http://localhost:8000`.

## Environment Variables

- `HF_TOKEN` - Hugging Face authentication token (required for model downloads)
- `KIE_API_KEY` - kie.ai API key for image generation
- `KIE_API_ENDPOINT` - kie.ai API endpoint URL (optional, defaults to placeholder)
- `PORT` - Server port (default: 8000)
- `HOST` - Server host (default: 0.0.0.0)

## API Endpoints

### Health Check

```bash
GET /api/health
```

Returns the status of loaded models.

### Process Image

```bash
POST /api/process/image
Content-Type: multipart/form-data

Parameters:
  - file: image file
  - text_prompt: text description of object to segment
```

### Process Video

```bash
POST /api/process/video
Content-Type: multipart/form-data

Parameters:
  - file: video file
  - text_prompt: text description of object to track
```

### Generate Image with kie.ai

```bash
POST /api/generate/image-kie
Content-Type: multipart/form-data

Parameters:
  - text_prompt: text description of image to generate
```

### Generate 3D Model

```bash
POST /api/generate/3d
Content-Type: multipart/form-data

Parameters:
  - file: image file
```

## Development

The server automatically serves the React frontend from the `frontend/dist` directory. To update the frontend:

```bash
cd frontend
npm install
npm run build
```

Then restart the server to serve the updated frontend.

## License

See the [LICENSE](../LICENSE) file in the root directory.
