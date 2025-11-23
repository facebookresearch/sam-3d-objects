import { useState, useEffect } from 'react'
import axios from 'axios'

interface ImageGeneratorProps {
  onGeneratedImage: (imageUrl: string) => void
}

export default function ImageGenerator({ onGeneratedImage }: ImageGeneratorProps) {
  const [prompt, setPrompt] = useState<string>('')
  const [loading, setLoading] = useState(false)
  const [generatedImage, setGeneratedImage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Clean up object URL on unmount or when it changes
  useEffect(() => {
    return () => {
      if (generatedImage && generatedImage.startsWith('blob:')) {
        URL.revokeObjectURL(generatedImage)
      }
    }
  }, [generatedImage])

  const handleGenerate = async () => {
    if (!prompt) {
      setError('Please enter a prompt')
      return
    }

    setLoading(true)
    setError(null)
    setGeneratedImage(null)

    try {
      const formData = new FormData()
      formData.append('text_prompt', prompt)

      const response = await axios.post('/api/generate/image-kie', formData, {
        responseType: 'blob',
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })

      // Create a URL for the generated image
      const imageUrl = URL.createObjectURL(response.data)
      setGeneratedImage(imageUrl)
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to generate image')
    } finally {
      setLoading(false)
    }
  }

  const handleMake3D = () => {
    if (generatedImage) {
      onGeneratedImage(generatedImage)
    }
  }

  return (
    <div className="section">
      <h2>Image Generator</h2>
      <p>Generate images using kie.ai's "Nano Banana" service</p>

      <div className="form-group">
        <label htmlFor="image-prompt">Text Prompt</label>
        <textarea
          id="image-prompt"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Describe the image you want to generate..."
          rows={4}
        />
      </div>

      <button
        className="primary"
        onClick={handleGenerate}
        disabled={loading || !prompt}
      >
        {loading ? 'Generating...' : 'Generate with Nano Banana'}
      </button>

      {error && <div className="error">{error}</div>}

      {loading && (
        <div className="loading">
          <p>Generating your image... This may take a moment.</p>
        </div>
      )}

      {generatedImage && (
        <div className="result-area">
          <h3>Generated Image</h3>
          <img
            src={generatedImage}
            alt="Generated"
            className="preview-image"
          />
          
          <div style={{ marginTop: '1rem' }}>
            <button className="primary" onClick={handleMake3D}>
              Make this 3D
            </button>
            <a href={generatedImage} download="generated_image.png">
              <button className="secondary">Download Image</button>
            </a>
          </div>
        </div>
      )}
    </div>
  )
}
