import React, { useState } from 'react';

function App() {
  // State for base language
  const [baseLanguage, setBaseLanguage] = useState('');

  // State for gender selection
  const [gender, setGender] = useState('');

  const baseLanguageMapping = {
    english: 'en-US',
    spanish: 'es-US',
    korean: 'ko-KR',
  };

  const handleConfirm = () => {
    if (!baseLanguage) {
      alert('Please select a base language.');
      return;
    }
    if (!gender) {
      alert('Please select a gender.');
      return;
    }

    const settings = {
      baseLanguage: baseLanguageMapping[baseLanguage],
      gender: gender,
    };

    // Send settings to the backend
    fetch('http://localhost:5000/set_settings', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(settings),
    })
      .then((response) => response.json())
      .then((data) => {
        console.log('Settings updated:', data);
      })
      .catch((error) => {
        console.error('Error updating settings:', error);
      });
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial' }}>
      <h1>Settings</h1>

      {/* Base Language Selection Dropdown */}
      <div style={{ marginBottom: '20px' }}>
        <label htmlFor="language-select">Base Language:</label><br />
        <select
          id="language-select"
          value={baseLanguage}
          onChange={(e) => setBaseLanguage(e.target.value)}
          style={{ width: '200px', padding: '5px', marginTop: '5px' }}
        >
          <option value="">Select Base Language</option>
          <option value="english">English</option>
          <option value="spanish">Spanish</option>
          <option value="korean">Korean</option>
        </select>
      </div>

      {/* Gender Selection Dropdown */}
      <div style={{ marginBottom: '20px' }}>
        <label htmlFor="gender-select">Select Gender:</label><br />
        <select
          id="gender-select"
          value={gender}
          onChange={(e) => setGender(e.target.value)}
          style={{ width: '200px', padding: '5px', marginTop: '5px' }}
        >
          <option value="">Select Gender</option>
          <option value="FEMALE">Female</option>
          <option value="MALE">Male</option>
        </select>
      </div>

      {/* Confirm Button */}
      <button
        onClick={handleConfirm}
        style={{
          padding: '10px 20px',
          backgroundColor: '#6200ee',
          color: '#fff',
          border: 'none',
          cursor: 'pointer',
        }}
      >
        Confirm
      </button>
    </div>
  );
}

export default App;
