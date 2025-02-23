import React, { useState } from 'react';

function App() {
  const [baseLanguage, setBaseLanguage] = useState('');
  const [gender, setGender] = useState('');

  const languageMapping = {
    english: 'en-US',
    spanish: 'es-US',
    korean: 'ko-KR',
  };

  const handleConfirm = async () => {
    if (!baseLanguage || !gender) {
      alert('Please fill in all fields.');
      return;
    }

    const settings = {
      baseLanguage: languageMapping[baseLanguage],
      gender,
    };

    try {
      // Proxying through Netlify if _redirects is set up
      const response = await fetch('/api/set_settings', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(settings),
      });

      const data = await response.json();
      console.log('Settings updated:', data);
      alert('Settings updated successfully.');
    } catch (error) {
      console.error('Error updating settings:', error);
      alert('Error updating settings. Please try again.');
    }
  };

  return (
    <div style={{
      backgroundColor: '#000000',
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      color: '#FFFFFF'
    }}>
      <div style={{
        width: '350px',
        padding: '30px',
        borderRadius: '8px',
        backgroundColor: '#1a1a1a',
        boxShadow: '0 4px 8px rgba(0,0,0,0.2)',
        textAlign: 'center'
      }}>
        <h1 style={{ marginBottom: '20px', color: '#F77F00' }}>Translator Settings</h1>
        <div style={{ marginBottom: '20px' }}>
          <label htmlFor="base-language" style={{ display: 'block', marginBottom: '8px' }}>Base Language:</label>
          <select
            id="base-language"
            value={baseLanguage}
            onChange={(e) => setBaseLanguage(e.target.value)}
            style={{
              width: '100%',
              padding: '10px',
              borderRadius: '4px',
              border: '1px solid #F77F00',
              backgroundColor: '#000000',
              color: '#FFFFFF'
            }}
          >
            <option value="">Select Base Language</option>
            <option value="english">English</option>
            <option value="spanish">Spanish</option>
            <option value="korean">Korean</option>
          </select>
        </div>

        <div style={{ marginBottom: '20px' }}>
          <label htmlFor="gender" style={{ display: 'block', marginBottom: '8px' }}>Voice Gender:</label>
          <select
            id="gender"
            value={gender}
            onChange={(e) => setGender(e.target.value)}
            style={{
              width: '100%',
              padding: '10px',
              borderRadius: '4px',
              border: '1px solid #F77F00',
              backgroundColor: '#000000',
              color: '#FFFFFF'
            }}
          >
            <option value="">Select Gender</option>
            <option value="FEMALE">Female</option>
            <option value="MALE">Male</option>
          </select>
        </div>

        <button
          onClick={handleConfirm}
          style={{
            width: '100%',
            padding: '12px',
            border: 'none',
            borderRadius: '4px',
            backgroundColor: '#D62828',
            color: '#FFFFFF',
            fontSize: '16px',
            cursor: 'pointer'
          }}
        >
          Confirm
        </button>
      </div>
    </div>
  );
}

export default App;
