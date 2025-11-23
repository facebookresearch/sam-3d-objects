import { useState, useCallback, useEffect, Suspense } from 'react'
import axios from 'axios'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, PerspectiveCamera, Environment } from '@react-three/drei'
import * as THREE from 'three'
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js'

interface ThreeDCreatorProps {
  generatedImage: string | null
}

function Model3D({ modelUrl }: { modelUrl: string }) {
  const [geometry, setGeometry] = useState<THREE.BufferGeometry | null>(null)

  useEffect(() => {
    const loader = new PLYLoader()
    loader.load(
      modelUrl,
      (geometry: THREE.BufferGeometry) => {
        geometry.computeVertexNormals()
        setGeometry(geometry)
      },
      undefined,
      (error: unknown) => {
        console.error('Error loading PLY:', error)
      }
    )
  }, [modelUrl])

  if (!geometry) return null

  return (
    <mesh geometry={geometry}>
      <meshStandardMaterial vertexColors side={THREE.DoubleSide} />
    </mesh>
  )
}

export default function ThreeDCreator({ generatedImage }: ThreeDCreatorProps) {
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string>(generatedImage || '')
  const [loading, setLoading] = useState(false)
  const [model3D, setModel3D] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Update preview when generatedImage changes
  useEffect(() => {
    if (generatedImage) {
      setPreviewUrl(generatedImage)
      // Convert blob URL to File object for upload
      fetch(generatedImage)
        .then(res => res.blob())
        .then(blob => {
          const file = new File([blob], 'generated_image.png', { type: 'image/png' })
          setFile(file)
        })
        .catch(err => console.error('Error converting image:', err))
    }
  }, [generatedImage])

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (selectedFile) {
      setFile(selectedFile)
      setPreviewUrl(URL.createObjectURL(selectedFile))
      setModel3D(null)
      setError(null)
    }
  }, [])

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    
    const droppedFile = e.dataTransfer.files[0]
    if (droppedFile && droppedFile.type.startsWith('image/')) {
      setFile(droppedFile)
      setPreviewUrl(URL.createObjectURL(droppedFile))
      setModel3D(null)
      setError(null)
    }
  }, [])

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
  }, [])

  const handleGenerate3D = async () => {
    if (!file) {
      setError('Please upload an image')
      return
    }

    setLoading(true)
    setError(null)
    setModel3D(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await axios.post('/api/generate/3d', formData, {
        responseType: 'blob',
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })

      // Create a URL for the 3D model
      const modelUrl = URL.createObjectURL(response.data)
      setModel3D(modelUrl)
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to generate 3D model')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="section">
      <h2>3D Creator</h2>
      <p>Generate a 3D model from an image</p>

      {!generatedImage && (
        <>
          <div
            className="upload-area"
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onClick={() => document.getElementById('3d-file-input')?.click()}
          >
            {previewUrl ? (
              <div>
                <img src={previewUrl} alt="Preview" style={{ maxWidth: '400px' }} />
                <p>{file?.name}</p>
              </div>
            ) : (
              <div>
                <p>📁 Drop an image here, or click to browse</p>
                <p style={{ fontSize: '0.9rem', color: '#666' }}>
                  Supported formats: JPG, PNG
                </p>
              </div>
            )}
          </div>

          <input
            id="3d-file-input"
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            style={{ display: 'none' }}
          />
        </>
      )}

      {generatedImage && previewUrl && (
        <div className="result-area" style={{ marginBottom: '1.5rem' }}>
          <h3>Source Image</h3>
          <img
            src={previewUrl}
            alt="Source"
            className="preview-image"
            style={{ maxWidth: '400px' }}
          />
        </div>
      )}

      <button
        className="primary"
        onClick={handleGenerate3D}
        disabled={loading || !file}
      >
        {loading ? 'Generating 3D Model...' : 'Generate 3D Model'}
      </button>

      {error && <div className="error">{error}</div>}

      {loading && (
        <div className="loading">
          <p>Generating your 3D model... This may take a while.</p>
        </div>
      )}

      {model3D && (
        <div className="result-area">
          <h3>3D Model Viewer</h3>
          <div className="viewer-container">
            <Canvas>
              <PerspectiveCamera makeDefault position={[0, 0, 3]} />
              <OrbitControls enableDamping />
              <ambientLight intensity={0.5} />
              <directionalLight position={[10, 10, 5]} intensity={1} />
              <Environment preset="studio" />
              <Suspense fallback={null}>
                <Model3D modelUrl={model3D} />
              </Suspense>
            </Canvas>
          </div>
          
          <div style={{ marginTop: '1rem' }}>
            <a href={model3D} download="model.ply">
              <button className="secondary">Download 3D Model (.ply)</button>
            </a>
          </div>
        </div>
      )}
    </div>
  )
}
