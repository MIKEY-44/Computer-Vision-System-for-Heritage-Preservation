import React, { useState } from 'react';
import axios from 'axios';
import { FiSend, FiMessageCircle } from 'react-icons/fi';

function Chatbot({ apiUrl }) {
  const [messages, setMessages] = useState([
    {
      type: 'bot',
      content: 'Hello! 👋 I\'m your Pushkarani guide. Ask me anything about temple ponds - their types, history, engineering, conservation, or interesting facts!',
      timestamp: new Date()
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    // Add user message
    const userMessage = {
      type: 'user',
      content: inputValue,
      timestamp: new Date()
    };

    setMessages([...messages, userMessage]);
    setInputValue('');
    setLoading(true);

    try {
      const response = await axios.post(`${apiUrl}/chatbot`, {
        message: inputValue
      });

      const botResponse = response.data.bot_response;
      let botContent = botResponse.message;

      // Format different types of responses
      if (botResponse.type === 'type_info') {
        botContent = (
          <div className="chatbot-response">
            <h4>{botResponse.data.name}</h4>
            <p>{botResponse.data.definition}</p>
            {botResponse.data.characteristics && (
              <div>
                <strong>Characteristics:</strong>
                <ul>
                  {botResponse.data.characteristics.map((char, idx) => (
                    <li key={idx}>{char}</li>
                  ))}
                </ul>
              </div>
            )}
            {botResponse.data.example && <p><strong>Example:</strong> {botResponse.data.example}</p>}
          </div>
        );
      } else if (botResponse.type === 'facts') {
        botContent = (
          <div className="chatbot-response fact">
            <p>📚 {botResponse.data}</p>
          </div>
        );
      } else if (botResponse.suggestions) {
        botContent = (
          <div className="chatbot-response suggestions">
            <p>{botResponse.message}</p>
            <div className="suggestion-buttons">
              {botResponse.suggestions.map((suggestion, idx) => (
                <button
                  key={idx}
                  className="suggestion-btn"
                  onClick={() => {
                    setInputValue(suggestion);
                  }}
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        );
      }

      const botMessage = {
        type: 'bot',
        content: botContent,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      const errorMessage = {
        type: 'bot',
        content: '❌ Sorry, I encountered an error. Please try again.',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="chatbot">
      <div className="chatbot-header">
        <FiMessageCircle size={28} />
        <h2>Pushkarani Knowledge Assistant</h2>
      </div>

      <div className="chatbot-messages">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message message-${msg.type}`}>
            <div className="message-content">
              {typeof msg.content === 'string' ? (
                <p>{msg.content}</p>
              ) : (
                msg.content
              )}
            </div>
            <small className="message-time">
              {msg.timestamp.toLocaleTimeString()}
            </small>
          </div>
        ))}
        {loading && (
          <div className="message message-bot">
            <div className="message-content">
              <div className="typing-indicator">
                <span></span><span></span><span></span>
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="chatbot-input">
        <textarea
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Ask about Pushkarani types, history, facts..."
          disabled={loading}
          rows="2"
        />
        <button
          className="send-btn"
          onClick={handleSendMessage}
          disabled={loading || !inputValue.trim()}
        >
          <FiSend size={20} />
        </button>
      </div>
    </div>
  );
}

export default Chatbot;
