import { useState } from 'react'
import './App.css'
import ObjectTracker from './components/ObjectTracker'
import ImageGenerator from './components/ImageGenerator'
import ThreeDCreator from './components/ThreeDCreator'

function App() {
  const [activeTab, setActiveTab] = useState<'tracker' | 'generator' | '3d'>('tracker')
  const [generatedImage, setGeneratedImage] = useState<string | null>(null)

  const handleGeneratedImage = (imageUrl: string) => {
    setGeneratedImage(imageUrl)
    setActiveTab('3d')
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>SAM 3D Objects - Unified Web Application</h1>
        <nav className="tab-navigation">
          <button
            className={activeTab === 'tracker' ? 'active' : ''}
            onClick={() => setActiveTab('tracker')}
          >
            Object Tracker
          </button>
          <button
            className={activeTab === 'generator' ? 'active' : ''}
            onClick={() => setActiveTab('generator')}
          >
            Image Generator
          </button>
          <button
            className={activeTab === '3d' ? 'active' : ''}
            onClick={() => setActiveTab('3d')}
          >
            3D Creator
          </button>
        </nav>
      </header>

      <main className="app-main">
        {activeTab === 'tracker' && <ObjectTracker />}
        {activeTab === 'generator' && (
          <ImageGenerator onGeneratedImage={handleGeneratedImage} />
        )}
        {activeTab === '3d' && <ThreeDCreator generatedImage={generatedImage} />}
      </main>
    </div>
  )
}

export default App
