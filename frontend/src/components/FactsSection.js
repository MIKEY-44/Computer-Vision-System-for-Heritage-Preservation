import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { FiRefreshCw, FiBook } from 'react-icons/fi';

function FactsSection({ apiUrl }) {
  const [facts, setFacts] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchFacts();
  }, []);

  const fetchFacts = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${apiUrl}/facts?count=10`);
      setFacts(response.data.facts);
    } catch (error) {
      console.error('Error fetching facts:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="facts-section">
      <div className="facts-header">
        <FiBook size={32} />
        <h2>Fascinating Facts About Pushkaranis</h2>
        <p>Discover the wonders of sacred temple ponds</p>
      </div>

      <div className="facts-grid">
        {facts.map((fact, idx) => (
          <div key={idx} className="fact-card">
            <div className="fact-number">{idx + 1}</div>
            <p className="fact-text">{fact}</p>
          </div>
        ))}
      </div>

      <button className="btn btn-secondary" onClick={fetchFacts} disabled={loading}>
        <FiRefreshCw size={18} />
        {loading ? 'Loading...' : 'Get More Facts'}
      </button>

      <div className="facts-gallery">
        <h3>Visual Heritage of Pushkaranis</h3>
        <div className="gallery-grid">
          <div className="gallery-item">
            <div className="gallery-placeholder">
              <img src="https://via.placeholder.com/300x200/FF6B6B/FFFFFF?text=Type-1+Teppakulam" alt="Type 1" />
            </div>
            <h4>🎭 Type-1: Teppakulam</h4>
            <p>Float Festival Tank with Central Pavilion</p>
            <p className="example">Example: Vandiyur Mariamman, Madurai</p>
          </div>

          <div className="gallery-item">
            <div className="gallery-placeholder">
              <img src="https://via.placeholder.com/300x200/4ECDC4/FFFFFF?text=Type-2+Kalyani" alt="Type 2" />
            </div>
            <h4>🔷 Type-2: Kalyani</h4>
            <p>Geometric Stepped Tank with Complex Patterns</p>
            <p className="example">Example: Hampi Royal Enclosure</p>
          </div>

          <div className="gallery-item">
            <div className="gallery-placeholder">
              <img src="https://via.placeholder.com/300x200/45B7D1/FFFFFF?text=Type-3+Kunda" alt="Type 3" />
            </div>
            <h4>▭ Type-3: Kunda</h4>
            <p>Simple Rectangular Multi-Purpose Tank</p>
            <p className="example">Example: Haridra Nadhi, Mannargudi</p>
          </div>
        </div>
      </div>

      <div className="key-statistics">
        <h3>Key Statistics</h3>
        <div className="stats-grid">
          <div className="stat-item">
            <div className="stat-number">100,000+</div>
            <p>Temple ponds in India</p>
          </div>
          <div className="stat-item">
            <div className="stat-number">23</div>
            <p>Acres - Largest tank (Mannargudi)</p>
          </div>
          <div className="stat-item">
            <div className="stat-number">2000+</div>
            <p>Years of documented history</p>
          </div>
          <div className="stat-item">
            <div className="stat-number">7</div>
            <p>Levels deep - Rani ki Vav, UNESCO site</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default FactsSection;
