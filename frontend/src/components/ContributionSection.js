import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { FiGift, FiCheckCircle, FiHeart } from 'react-icons/fi';

function ContributionSection({ apiUrl }) {
  const [conservationData, setConservationData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchConservationInfo();
  }, []);

  const fetchConservationInfo = async () => {
    try {
      const response = await axios.get(`${apiUrl}/conservation`);
      setConservationData(response.data);
    } catch (error) {
      console.error('Error fetching conservation info:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) return <div>Loading...</div>;

  return (
    <div className="contribution-section">
      <div className="contribution-header">
        <FiGift size={32} />
        <h2>Contribute to Pushkarani Conservation</h2>
        <p>Be part of preserving India's sacred water heritage</p>
      </div>

      {conservationData && (
        <>
          <div className="critical-issue">
            <div className="issue-content">
              <h3>⚠️ The Challenge</h3>
              <div className="issue-stats">
                <div className="stat">
                  <div className="stat-number">{conservationData.critical_issue.split(/\d+/)[0]}100,000+</div>
                  <p>Temple ponds not well maintained</p>
                </div>
                <p className="issue-description">{conservationData.water_loss}</p>
              </div>
            </div>
          </div>

          <div className="how-to-contribute">
            <h3>🤝 How You Can Help</h3>
            <div className="contribution-methods">
              {conservationData.how_to_contribute.map((method, idx) => (
                <div key={idx} className="contribution-card">
                  <div className="card-number">{idx + 1}</div>
                  <div className="card-content">
                    <FiCheckCircle size={24} className="card-icon" />
                    <p>{method}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="impact-section">
            <FiHeart size={32} className="heart-icon" />
            <h3>Impact of Conservation</h3>
            <p className="impact-text">{conservationData.impact}</p>
            
            <div className="impact-areas">
              <div className="impact-item">
                <h4>🏛️ Cultural Heritage</h4>
                <p>Preserve ancient temples and architectural treasures</p>
              </div>
              <div className="impact-item">
                <h4>💧 Water Security</h4>
                <p>Ensure sustainable groundwater recharge and storage</p>
              </div>
              <div className="impact-item">
                <h4>🌍 Environment</h4>
                <p>Support local ecosystems and biodiversity</p>
              </div>
              <div className="impact-item">
                <h4>👥 Community</h4>
                <p>Create social gathering spaces and cultural venues</p>
              </div>
            </div>
          </div>

          <div className="upload-section">
            <h3>📸 Share Your Pushkarani Photos</h3>
            <p>Help us build a comprehensive database of temple ponds across India</p>
            <div className="upload-box-contribute">
              <input 
                type="file" 
                accept="image/*" 
                id="contribute-image"
                style={{ display: 'none' }}
              />
              <label htmlFor="contribute-image" className="upload-label-contribute">
                <span>📷 Click to Upload Image</span>
                <p>Include location details to help with documentation</p>
              </label>
            </div>
          </div>

          <div className="contact-section">
            <h3>🤓 More Information</h3>
            <p>For conservation projects, partnerships, or more information, contact:</p>
            <div className="contact-info">
              <div className="contact-item">
                <strong>Documentation Project</strong>
                <p>Document temple ponds in your region</p>
              </div>
              <div className="contact-item">
                <strong>Restoration Initiatives</strong>
                <p>Join local tank restoration and cleaning campaigns</p>
              </div>
              <div className="contact-item">
                <strong>Awareness Programs</strong>
                <p>Participate in educational and community programs</p>
              </div>
            </div>
          </div>
        </>
      )}

      <div className="contribution-cta">
        <h3>Ready to Make a Difference?</h3>
        <p>Every action counts in preserving our sacred heritage</p>
        <button className="btn btn-primary btn-large">Start Contributing Today</button>
      </div>
    </div>
  );
}

export default ContributionSection;
