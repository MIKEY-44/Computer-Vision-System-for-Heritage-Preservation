import React from 'react';
import { FiUploadCloud, FiX } from 'react-icons/fi';

function ImageUploader({ onImageSelect, selectedImage }) {
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      // Validate file type
      if (!file.type.startsWith('image/')) {
        alert('Please select an image file');
        return;
      }
      onImageSelect(file);
    }
  };

  const handleRemove = () => {
    onImageSelect(null);
  };

  return (
    <div className="image-uploader">
      <div className="upload-box">
        {selectedImage ? (
          <div className="image-preview">
            <img 
              src={URL.createObjectURL(selectedImage)} 
              alt="Selected"
            />
            <button className="remove-btn" onClick={handleRemove}>
              <FiX /> Remove
            </button>
            <p className="file-name">{selectedImage.name}</p>
          </div>
        ) : (
          <label className="upload-label">
            <FiUploadCloud size={48} />
            <p>Click or drag image here</p>
            <p className="sub-text">PNG, JPG, BMP, TIFF</p>
            <input
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              style={{ display: 'none' }}
            />
          </label>
        )}
      </div>
    </div>
  );
}

export default ImageUploader;
