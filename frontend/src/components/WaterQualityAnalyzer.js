import React from 'react';
import { FiTrendingUp, FiAlertCircle } from 'react-icons/fi';

function WaterQualityAnalyzer({ result }) {
  const getQualityColor = (quality) => {
    if (quality >= 80) return '#10B981'; // Green
    if (quality >= 60) return '#F59E0B'; // Amber
    if (quality >= 40) return '#F97316'; // Orange
    return '#EF4444'; // Red
  };

  const getQualityEmoji = (category) => {
    const emojis = {
      'Excellent': '✨',
      'Good': '👍',
      'Fair': '⚠️',
      'Poor': '❌',
      'Critical': '🚨'
    };
    return emojis[category] || '🔍';
  };

  return (
    <div className="water-quality-analyzer">
      <div className="quality-overview">
        <div className="quality-score">
          <div 
            className="score-circle"
            style={{ borderColor: getQualityColor(result.overall_quality) }}
          >
            <span className="score-value">{result.overall_quality.toFixed(1)}</span>
            <span className="score-label">/ 100</span>
          </div>
          <div className="quality-status">
            <h4>{getQualityEmoji(result.quality_category)} {result.quality_category}</h4>
            <p>Overall Water Quality</p>
          </div>
        </div>
      </div>

      <div className="quality-metrics">
        <h4>Detailed Metrics</h4>
        
        <div className="metric">
          <div className="metric-header">
            <span className="metric-name">Clarity Score</span>
            <span className="metric-value">{result.clarity_score.toFixed(1)}%</span>
          </div>
          <div className="metric-bar">
            <div 
              className="metric-fill"
              style={{ width: `${result.clarity_score}%`, backgroundColor: '#3B82F6' }}
            ></div>
          </div>
        </div>

        <div className="metric">
          <div className="metric-header">
            <span className="metric-name">Turbidity</span>
            <span className="metric-value">{result.turbidity_score.toFixed(1)}%</span>
          </div>
          <div className="metric-bar">
            <div 
              className="metric-fill"
              style={{ width: `${result.turbidity_score}%`, backgroundColor: '#8B5CF6' }}
            ></div>
          </div>
          <small>Lower is better</small>
        </div>

        <div className="metric">
          <div className="metric-header">
            <span className="metric-name">Algae Likelihood</span>
            <span className="metric-value">{result.algae_likelihood.toFixed(1)}%</span>
          </div>
          <div className="metric-bar">
            <div 
              className="metric-fill"
              style={{ width: `${result.algae_likelihood}%`, backgroundColor: '#EF4444' }}
            ></div>
          </div>
        </div>
      </div>

      <div className="color-analysis">
        <h4>Color Analysis</h4>
        <div className="color-breakdown">
          <div className="color-item">
            <div 
              className="color-sample"
              style={{ backgroundColor: `rgb(${result.color_analysis.red}, 0, 0)` }}
            ></div>
            <span>Red: {result.color_analysis.red.toFixed(1)}</span>
          </div>
          <div className="color-item">
            <div 
              className="color-sample"
              style={{ backgroundColor: `rgb(0, ${result.color_analysis.green}, 0)` }}
            ></div>
            <span>Green: {result.color_analysis.green.toFixed(1)}</span>
          </div>
          <div className="color-item">
            <div 
              className="color-sample"
              style={{ backgroundColor: `rgb(0, 0, ${result.color_analysis.blue})` }}
            ></div>
            <span>Blue: {result.color_analysis.blue.toFixed(1)}</span>
          </div>
        </div>
      </div>

      {result.recommendations && result.recommendations.length > 0 && (
        <div className="recommendations">
          <h4><FiAlertCircle /> Recommendations</h4>
          <ul>
            {result.recommendations.map((rec, idx) => (
              <li key={idx}>{rec}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="water-quality-info">
        <h5>Understanding Water Quality Metrics</h5>
        <ul>
          <li><strong>Clarity:</strong> How clear the water is (higher is better)</li>
          <li><strong>Turbidity:</strong> Amount of suspended particles (lower is better)</li>
          <li><strong>Algae Likelihood:</strong> Chance of algae bloom presence (lower is better)</li>
        </ul>
      </div>
    </div>
  );
}

export default WaterQualityAnalyzer;
