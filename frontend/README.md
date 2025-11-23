# SAM 3D Objects Frontend

React + TypeScript + Vite frontend for the SAM 3D Objects unified web application.

## Features

- **Object Tracker**: Upload images or videos and track objects with text prompts
- **Image Generator**: Generate images using kie.ai's Nano Banana service
- **3D Creator**: Convert images to 3D models with an interactive viewer

## Tech Stack

- React 19
- TypeScript
- Vite
- React Three Fiber (@react-three/fiber) for 3D visualization
- Axios for API communication

## Development

### Prerequisites

- Node.js 18+ and npm

### Installation

```bash
cd frontend
npm install
```

### Running Development Server

```bash
npm run dev
```

This will start the development server on `http://localhost:5173`.

The Vite dev server is configured to proxy API requests to `http://localhost:8000`, so make sure the backend server is running.

### Building for Production

```bash
npm run build
```

This creates an optimized production build in the `dist/` directory.

### Preview Production Build

```bash
npm run preview
```

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── ObjectTracker.tsx    # Image/video processing component
│   │   ├── ImageGenerator.tsx   # Image generation component
│   │   └── ThreeDCreator.tsx    # 3D model viewer component
│   ├── App.tsx                   # Main app with navigation
│   ├── App.css                   # App styles
│   ├── index.css                 # Global styles
│   └── main.tsx                  # Entry point
├── public/                       # Static assets
├── dist/                         # Production build (generated)
├── package.json
├── tsconfig.json
└── vite.config.ts
```

## API Integration

The frontend communicates with the backend using relative paths:

- `/api/process/image` - Process images
- `/api/process/video` - Process videos
- `/api/generate/image-kie` - Generate images with kie.ai
- `/api/generate/3d` - Generate 3D models

This ensures compatibility with the unified serving model where the backend serves both API and frontend.

## Components

### ObjectTracker

Allows users to upload images or videos and specify objects to track using text prompts. Results are returned as PLY files.

### ImageGenerator

Uses kie.ai's Nano Banana service to generate images from text prompts. Includes a "Make this 3D" button to transition to the 3D Creator.

### ThreeDCreator

Converts images to 3D models and displays them in an interactive viewer using React Three Fiber. Supports drag-and-drop, file upload, and viewing PLY format models.

## License

See the [LICENSE](../LICENSE) file in the root directory.

