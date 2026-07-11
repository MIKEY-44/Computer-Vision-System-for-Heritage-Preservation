import React from 'react';
import { FiCheckCircle, FiInfo } from 'react-icons/fi';

function ResultsDisplay({ result }) {
  // Safely handle both single model and ensemble responses
  if (!result) {
    return <div className="results-display"><p>No results available</p></div>;
  }

  // Get prediction class from either predicted_class (single model) or final_prediction (ensemble)
  const predictedClass = result.predicted_class || result.final_prediction;
  
  // Get confidence from either confidence or average_confidence
  const confidence = result.confidence || result.average_confidence || 0;
  
  // Get all probabilities or confidence_by_class
  const allProbs = result.all_probabilities || result.confidence_by_class || {};

  if (!predictedClass) {
    return <div className="results-display"><p>Error: No prediction class found</p></div>;
  }

  const getTypeColor = (typeName) => {
    switch (typeName) {
      case 'type-1':
        return '#FF6B6B';
      case 'type-2':
        return '#4ECDC4';
      case 'type-3':
        return '#45B7D1';
      default:
        return '#666';
    }
  };

  const getTypeIcon = (typeName) => {
    const icons = {
      'type-1': '🎭',
      'type-2': '🔷',
      'type-3': '▭'
    };
    return icons[typeName] || '🏛️';
  };

  return (
    <div className="results-display">
      <div className="results-header">
        <FiCheckCircle size={32} color="#10B981" />
        <h3>Classification Results</h3>
      </div>

      <div className="main-prediction">
        <div 
          className="prediction-box"
          style={{ borderLeftColor: getTypeColor(predictedClass) }}
        >
          <div className="prediction-header">
            <span className="type-icon">{getTypeIcon(predictedClass)}</span>
            <div>
              <h4>Predicted Type</h4>
              <p className="type-name">{String(predictedClass).toUpperCase()}</p>
            </div>
          </div>
          
          <div className="confidence-section">
            <p className="confidence-label">Confidence Level</p>
            <div className="confidence-bar">
              <div 
                className="confidence-fill"
                style={{ width: `${confidence}%` }}
              ></div>
            </div>
            <p className="confidence-value">{Number(confidence).toFixed(2)}%</p>
          </div>

          {result.class_descriptions && (
            <div className="description">
              <h5>{result.class_descriptions.name}</h5>
              <p>{result.class_descriptions.definition}</p>
              {result.class_descriptions.characteristics && (
                <div className="characteristics">
                  <p><strong>Key Characteristics:</strong></p>
                  <ul>
                    {result.class_descriptions.characteristics.map((char, idx) => (
                      <li key={idx}>{char}</li>
                    ))}
                  </ul>
                </div>
              )}
              {result.class_descriptions.example && (
                <p><strong>Example:</strong> {result.class_descriptions.example}</p>
              )}
              {result.class_descriptions.patronage && (
                <p><strong>Patronage:</strong> {result.class_descriptions.patronage}</p>
              )}
            </div>
          )}
        </div>
      </div>

      <div className="probabilities">
        <h4>All Predictions</h4>
        <div className="probability-list">
          {Object.entries(allProbs).map(([className, probability]) => (
            <div key={className} className="probability-item">
              <span className="class-label">{String(className).toUpperCase()}</span>
              <div className="probability-bar">
                <div 
                  className="probability-fill"
                  style={{ width: `${Number(probability)}%` }}
                ></div>
              </div>
              <span className="probability-value">{Number(probability).toFixed(2)}%</span>
            </div>
          ))}
        </div>
      </div>

      {result.method === 'ensemble' && (
        <div className="ensemble-info">
          <FiInfo size={20} />
          <p>
            This result was generated using {result.num_models} different models:
            {' '}{Array.isArray(result.models_used) ? result.models_used.map(m => String(m).toUpperCase()).join(', ') : 'Multiple models'}
          </p>
        </div>
      )}

      {/* Stone Quality Assessment */}
      {result.stone_quality_assessment && (
        <div className="stone-quality-section">
          <h4>Stone Quality Assessment</h4>
          <div className="quality-box">
            <div className="quality-header">
              <h5>Structural Quality: <span className={`quality-${String(result.stone_quality_assessment.quality).toLowerCase()}`}>
                {String(result.stone_quality_assessment.quality).toUpperCase()}
              </span></h5>
              <p className="quality-confidence">
                Confidence: {Number(result.stone_quality_assessment.confidence).toFixed(2)}%
              </p>
            </div>
            
            <div className="quality-bar">
              <div 
                className="quality-fill"
                style={{ width: `${Number(result.stone_quality_assessment.confidence)}%` }}
              ></div>
            </div>

            {result.stone_quality_assessment.maintenance_required && (
              <div className="maintenance-alert">
                <p>⚠️ Maintenance Required</p>
                <p>Urgency Level: <strong>{String(result.stone_quality_assessment.maintenance_urgency).toUpperCase()}</strong></p>
              </div>
            )}

            {result.stone_quality_assessment.all_predictions && (
              <div className="quality-predictions">
                <p><strong>Quality Breakdown:</strong></p>
                <div className="quality-breakdown">
                  {Object.entries(result.stone_quality_assessment.all_predictions).map(([quality, score]) => (
                    <div key={quality} className="quality-item">
                      <span>{String(quality)}:</span>
                      <span>{Number(score).toFixed(2)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Show ensemble model breakdown if available */}
      {result.detailed_breakdown && (
        <div className="ensemble-breakdown">
          <h4>Individual Model Predictions</h4>
          {Object.entries(result.detailed_breakdown).map(([className, data]) => (
            <div key={className} className="ensemble-class">
              <h5>{String(className).toUpperCase()}</h5>
              <p>Average: {Number(data.average_confidence || 0).toFixed(2)}%</p>
              {data.individual_model_predictions && (
                <div className="model-predictions">
                  {Object.entries(data.individual_model_predictions).map(([modelName, pred]) => (
                    <div key={modelName} className="model-pred">
                      <span>{String(modelName).toUpperCase()}:</span>
                      <span>{Number(pred).toFixed(2)}%</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      <div className="timestamp">
        <small>Prediction made at: {new Date(result.timestamp).toLocaleString()}</small>
      </div>
    </div>
  );
}

export default ResultsDisplay;
