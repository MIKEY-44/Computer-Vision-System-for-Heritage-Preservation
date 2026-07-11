import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { FiUpload, FiImage, FiZap, FiMessageCircle, FiInfo, FiGift } from 'react-icons/fi';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import './App.css';
import ImageUploader from './components/ImageUploader';
import ResultsDisplay from './components/ResultsDisplay';
import WaterQualityAnalyzer from './components/WaterQualityAnalyzer';
import Chatbot from './components/Chatbot';
import FactsSection from './components/FactsSection';
import ContributionSection from './components/ContributionSection';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

function App() {
  const [activeTab, setActiveTab] = useState('classify');
  const [selectedImage, setSelectedImage] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);
  const [waterQualityResult, setWaterQualityResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState('densenet');
  const [availableModels, setAvailableModels] = useState([]);
  const [useEnsemble, setUseEnsemble] = useState(false);

  // Load available models on mount
  useEffect(() => {
    fetchAvailableModels();
  }, []);

  const fetchAvailableModels = async () => {
    try {
      const response = await axios.get(`${API_URL}/models`);
      setAvailableModels(response.data.available_models);
    } catch (error) {
      console.error('Error fetching models:', error);
      toast.error('Failed to load models');
    }
  };

  const handleImageSelect = (file) => {
    setSelectedImage(file);
    setPredictionResult(null);
    setWaterQualityResult(null);
  };

  const handlePrediction = async () => {
    if (!selectedImage) {
      toast.warn('Please select an image first');
      return;
    }

    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('image', selectedImage);

      let response;
      if (useEnsemble) {
        response = await axios.post(`${API_URL}/predict/ensemble`, formData);
      } else {
        response = await axios.post(
          `${API_URL}/predict?model=${selectedModel}`,
          formData
        );
      }

      setPredictionResult(response.data);
      toast.success('Classification successful!');
    } catch (error) {
      console.error('Prediction error:', error);
      toast.error('Classification failed: ' + (error.response?.data?.error || error.message));
    } finally {
      setLoading(false);
    }
  };

  const handleWaterQualityAnalysis = async () => {
    if (!selectedImage) {
      toast.warn('Please select an image first');
      return;
    }

    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('image', selectedImage);

      const response = await axios.post(`${API_URL}/water-quality`, formData);
      setWaterQualityResult(response.data.water_quality);
      toast.success('Water quality analysis complete!');
    } catch (error) {
      console.error('Water quality analysis error:', error);
      toast.error('Analysis failed: ' + (error.response?.data?.error || error.message));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <h1>🏛️ Pushkarani Analysis System</h1>
          <p>Sacred Temple Ponds Classification & Water Quality Analysis</p>
        </div>
      </header>

      <nav className="nav-tabs">
        <button
          className={`nav-tab ${activeTab === 'classify' ? 'active' : ''}`}
          onClick={() => setActiveTab('classify')}
        >
          <FiImage /> Classify
        </button>
        <button
          className={`nav-tab ${activeTab === 'water-quality' ? 'active' : ''}`}
          onClick={() => setActiveTab('water-quality')}
        >
          <FiZap /> Water Quality
        </button>
        <button
          className={`nav-tab ${activeTab === 'chatbot' ? 'active' : ''}`}
          onClick={() => setActiveTab('chatbot')}
        >
          <FiMessageCircle /> Chatbot
        </button>
        <button
          className={`nav-tab ${activeTab === 'facts' ? 'active' : ''}`}
          onClick={() => setActiveTab('facts')}
        >
          <FiInfo /> Facts
        </button>
        <button
          className={`nav-tab ${activeTab === 'contribute' ? 'active' : ''}`}
          onClick={() => setActiveTab('contribute')}
        >
          <FiGift /> Contribute
        </button>
      </nav>

      <main className="main-content">
        {/* Classification Tab */}
        {activeTab === 'classify' && (
          <div className="tab-content">
            <div className="container">
              <div className="section">
                <h2>Temple Pond Classification</h2>
                <p>Upload an image to classify the Pushkarani type using AI models</p>

                <ImageUploader onImageSelect={handleImageSelect} selectedImage={selectedImage} />

                <div className="controls-section">
                  <div className="control-group">
                    <label htmlFor="model-select">Select Model:</label>
                    <select
                      id="model-select"
                      value={selectedModel}
                      onChange={(e) => setSelectedModel(e.target.value)}
                      disabled={useEnsemble || loading}
                      className="select-input"
                    >
                      {availableModels.map((model) => (
                        <option key={model} value={model}>
                          {model.toUpperCase()}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div className="control-group">
                    <label className="checkbox-label">
                      <input
                        type="checkbox"
                        checked={useEnsemble}
                        onChange={(e) => setUseEnsemble(e.target.checked)}
                        disabled={loading}
                      />
                      Use Ensemble Method (All Models)
                    </label>
                  </div>

                  <button
                    className="btn btn-primary"
                    onClick={handlePrediction}
                    disabled={loading || !selectedImage}
                  >
                    {loading ? 'Classifying...' : 'Classify Image'}
                  </button>
                </div>

                {predictionResult && <ResultsDisplay result={predictionResult} />}
              </div>
            </div>
          </div>
        )}

        {/* Water Quality Tab */}
        {activeTab === 'water-quality' && (
          <div className="tab-content">
            <div className="container">
              <div className="section">
                <h2>Water Quality Analysis</h2>
                <p>Analyze water clarity, algae presence, and overall quality</p>

                <ImageUploader onImageSelect={handleImageSelect} selectedImage={selectedImage} />

                <div className="controls-section">
                  <button
                    className="btn btn-primary"
                    onClick={handleWaterQualityAnalysis}
                    disabled={loading || !selectedImage}
                  >
                    {loading ? 'Analyzing...' : 'Analyze Water Quality'}
                  </button>
                </div>

                {waterQualityResult && <WaterQualityAnalyzer result={waterQualityResult} />}
              </div>
            </div>
          </div>
        )}

        {/* Chatbot Tab */}
        {activeTab === 'chatbot' && (
          <div className="tab-content">
            <div className="container">
              <Chatbot apiUrl={API_URL} />
            </div>
          </div>
        )}

        {/* Facts Tab */}
        {activeTab === 'facts' && (
          <div className="tab-content">
            <div className="container">
              <FactsSection apiUrl={API_URL} />
            </div>
          </div>
        )}

        {/* Contribution Tab */}
        {activeTab === 'contribute' && (
          <div className="tab-content">
            <div className="container">
              <ContributionSection apiUrl={API_URL} />
            </div>
          </div>
        )}
      </main>

      <footer className="footer">
        <p>🏛️ Pushkarani Classification System | Building AI for Sacred Heritage Preservation</p>
      </footer>

      <ToastContainer position="bottom-right" autoClose={3000} />
    </div>
  );
}

export default App;
