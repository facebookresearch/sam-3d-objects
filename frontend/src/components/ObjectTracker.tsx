import { useState, useCallback, useEffect } from 'react'
import axios from 'axios'

export default function ObjectTracker() {
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string>('')
  const [prompt, setPrompt] = useState<string>('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [fileType, setFileType] = useState<'image' | 'video' | null>(null)

  // Clean up object URLs on unmount or when they change
  useEffect(() => {
    return () => {
      if (previewUrl && previewUrl.startsWith('blob:')) {
        URL.revokeObjectURL(previewUrl)
      }
    }
  }, [previewUrl])

  useEffect(() => {
    return () => {
      if (result && result.startsWith('blob:')) {
        URL.revokeObjectURL(result)
      }
    }
  }, [result])

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (selectedFile) {
      // Revoke previous URL before creating new one
      if (previewUrl && previewUrl.startsWith('blob:')) {
        URL.revokeObjectURL(previewUrl)
      }
      
      setFile(selectedFile)
      setPreviewUrl(URL.createObjectURL(selectedFile))
      
      // Determine file type
      if (selectedFile.type.startsWith('image/')) {
        setFileType('image')
      } else if (selectedFile.type.startsWith('video/')) {
        setFileType('video')
      } else {
        setFileType(null)
      }
      
      setResult(null)
      setError(null)
    }
  }, [previewUrl])

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    
    const droppedFile = e.dataTransfer.files[0]
    if (droppedFile) {
      // Revoke previous URL before creating new one
      if (previewUrl && previewUrl.startsWith('blob:')) {
        URL.revokeObjectURL(previewUrl)
      }
      
      setFile(droppedFile)
      setPreviewUrl(URL.createObjectURL(droppedFile))
      
      if (droppedFile.type.startsWith('image/')) {
        setFileType('image')
      } else if (droppedFile.type.startsWith('video/')) {
        setFileType('video')
      } else {
        setFileType(null)
      }
      
      setResult(null)
      setError(null)
    }
  }, [previewUrl])

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
  }, [])

  const handleProcess = async () => {
    if (!file || !prompt) {
      setError('Please upload a file and enter a prompt')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('text_prompt', prompt)

      const endpoint = fileType === 'image' ? '/api/process/image' : '/api/process/video'
      
      const response = await axios.post(endpoint, formData, {
        responseType: 'blob',
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })

      // Create a URL for the result
      const resultUrl = URL.createObjectURL(response.data)
      setResult(resultUrl)
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to process file')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="section">
      <h2>Object Tracker</h2>
      <p>Upload an image or video and specify what object to track</p>

      <div
        className="upload-area"
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onClick={() => document.getElementById('file-input')?.click()}
      >
        {previewUrl ? (
          <div>
            {fileType === 'image' ? (
              <img src={previewUrl} alt="Preview" style={{ maxWidth: '400px' }} />
            ) : (
              <video src={previewUrl} controls style={{ maxWidth: '400px' }} />
            )}
            <p>{file?.name}</p>
          </div>
        ) : (
          <div>
            <p>📁 Drop an image or video here, or click to browse</p>
            <p style={{ fontSize: '0.9rem', color: '#666' }}>
              Supported formats: JPG, PNG, MP4, MOV
            </p>
          </div>
        )}
      </div>

      <input
        id="file-input"
        type="file"
        accept="image/*,video/*"
        onChange={handleFileChange}
        style={{ display: 'none' }}
      />

      <div className="form-group">
        <label htmlFor="prompt">Text Prompt</label>
        <input
          id="prompt"
          type="text"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="e.g., 'chair', 'person', 'car'"
        />
      </div>

      <button
        className="primary"
        onClick={handleProcess}
        disabled={loading || !file || !prompt}
      >
        {loading ? 'Processing...' : 'Process'}
      </button>

      {error && <div className="error">{error}</div>}

      {loading && (
        <div className="loading">
          <p>Processing your {fileType}... This may take a while.</p>
        </div>
      )}

      {result && (
        <div className="result-area">
          <h3>Result</h3>
          <p>Download the processed result:</p>
          <a href={result} download="processed_result.ply">
            <button className="secondary">Download Result (.ply)</button>
          </a>
        </div>
      )}
    </div>
  )
}
