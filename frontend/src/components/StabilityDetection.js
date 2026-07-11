import React from 'react';
import { FiAlertCircle, FiCheckCircle, FiAlertTriangle } from 'react-icons/fi';

function StabilityDetection({ stabilityData }) {
  if (!stabilityData) {
    return null;
  }

  // Get stability status from the actual model prediction if available
  const stabilityPrediction = stabilityData.stability_prediction;
  const status = stabilityPrediction?.status?.toLowerCase() || 'unknown';
  const confidence = stabilityPrediction?.confidence || 0;
  
  // Get maintenance recommendations from the old logic
  const maintenance_level = stabilityData.maintenance_level || '';
  const description = stabilityData.description || '';
  const recommendations = stabilityData.recommendations || [];
  const pollution_risk = stabilityData.pollution_risk || '';

  const getStabilityIcon = (stat) => {
    switch (stat) {
      case 'good':
        return <FiCheckCircle size={40} style={{ color: '#10B981' }} />;
      case 'medium':
        return <FiAlertTriangle size={40} style={{ color: '#F59E0B' }} />;
      case 'bad':
        return <FiAlertCircle size={40} style={{ color: '#EF4444' }} />;
      default:
        return null;
    }
  };

  const getStatusColor = (stat) => {
    switch (stat) {
      case 'good':
        return '#10B981';
      case 'medium':
        return '#F59E0B';
      case 'bad':
        return '#EF4444';
      default:
        return '#666';
    }
  };

  const getStatusBgColor = (stat) => {
    switch (stat) {
      case 'good':
        return 'rgba(16, 185, 129, 0.1)';
      case 'medium':
        return 'rgba(245, 158, 11, 0.1)';
      case 'bad':
        return 'rgba(239, 68, 68, 0.1)';
      default:
        return '#f5f5f5';
    }
  };

  return (
    <div className="stability-detection">
      <div className="stability-header">
        <h3>🌊 Water Stability Detection (Dataset1 Model)</h3>
        <p>AI-based water condition assessment and maintenance recommendations</p>
      </div>

      <div
        className="stability-status-card"
        style={{
          borderLeftColor: getStatusColor(status),
          backgroundColor: getStatusBgColor(status),
        }}
      >
        <div className="stability-status-header">
          <div className="stability-icon-container">
            {getStabilityIcon(status)}
          </div>
          <div className="stability-status-info">
            <h4>Water Stability: {String(status).toUpperCase()}</h4>
            <div className="confidence-bar-stability">
              <div
                className="confidence-fill-stability"
                style={{
                  width: `${confidence}%`,
                  backgroundColor: getStatusColor(status),
                }}
              ></div>
            </div>
            <p className="confidence-value-stability">{Number(confidence).toFixed(2)}% Confidence</p>
          </div>
        </div>
      </div>

      <div className="stability-details">
        {/* Status Description */}
        <div className="stability-section">
          <h4>Status Description</h4>
          <p className="description-text">{description}</p>
        </div>

        {/* Maintenance Level */}
        <div className="stability-section">
          <h4>Maintenance Level</h4>
          <div
            className="maintenance-badge"
            style={{
              backgroundColor: getStatusBgColor(status),
              borderLeftColor: getStatusColor(status),
            }}
          >
            <strong>{maintenance_level.toUpperCase()}</strong>
          </div>
        </div>

        {/* Pollution Risk */}
        {pollution_risk && (
          <div className="stability-section">
            <h4>⚠️ Pollution Risk Assessment</h4>
            <div className="pollution-risk-box">
              <p>{pollution_risk}</p>
            </div>
          </div>
        )}

        {/* Recommendations */}
        {recommendations && recommendations.length > 0 && (
          <div className="stability-section">
            <h4>📋 Recommended Actions</h4>
            <div className="recommendations-list">
              {recommendations.map((rec, idx) => (
                <div key={idx} className="recommendation-item">
                  <span className="rec-icon">✓</span>
                  <p>{rec}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* All Predictions from Model */}
        {stabilityPrediction?.all_predictions && (
          <div className="stability-section">
            <h4>📊 Model Confidence by Status</h4>
            <div className="model-confidence-breakdown">
              {Object.entries(stabilityPrediction.all_predictions).map(([stat, conf]) => (
                <div key={stat} className="confidence-item-stability">
                  <span className="status-label">{String(stat).toUpperCase()}</span>
                  <div className="confidence-bar-mini">
                    <div
                      className="confidence-fill-mini"
                      style={{
                        width: `${conf}%`,
                        backgroundColor: getStatusColor(stat),
                      }}
                    ></div>
                  </div>
                  <span className="confidence-val">{Number(conf).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default StabilityDetection;
