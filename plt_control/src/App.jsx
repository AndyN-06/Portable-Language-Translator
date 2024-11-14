import React, { useState } from 'react';

function App() {
  // State for base language
  const [baseLanguage, setBaseLanguage] = useState('');

  // State variables for voice settings
  const [englishVoiceSetting, setEnglishVoiceSetting] = useState('');
  const [spanishVoiceSetting, setSpanishVoiceSetting] = useState('');
  const [koreanVoiceSetting, setKoreanVoiceSetting] = useState('');

  const handleConfirm = () => {
    console.log('Selected Base Language:', baseLanguage);
    console.log('Selected English Voice Setting:', englishVoiceSetting);
    console.log('Selected Spanish Voice Setting:', spanishVoiceSetting);
    console.log('Selected Korean Voice Setting:', koreanVoiceSetting);

    // Prepare settings object
    const settings = {
      baseLanguage,
      englishVoiceSetting,
      spanishVoiceSetting,
      koreanVoiceSetting,
    };

    // Send settings to your API or another program
    // fetch('http://your-api-endpoint.com/settings', { ... })
  };

  const gcpVoiceOptions = [
    // **English (United Kingdom)**
    {
      languageCode: 'en-GB',
      voices: [
        { name: 'en-GB-Standard-A', gender: 'Female', type: 'Standard' },
        { name: 'en-GB-Standard-B', gender: 'Male', type: 'Standard' },
        { name: 'en-GB-Standard-C', gender: 'Female', type: 'Standard' },
        { name: 'en-GB-Standard-D', gender: 'Male', type: 'Standard' },
        { name: 'en-GB-Wavenet-A', gender: 'Female', type: 'WaveNet' },
        { name: 'en-GB-Wavenet-B', gender: 'Male', type: 'WaveNet' },
        { name: 'en-GB-Wavenet-C', gender: 'Female', type: 'WaveNet' },
        { name: 'en-GB-Wavenet-D', gender: 'Male', type: 'WaveNet' },
      ],
    },

    // **English (Australia)**
    {
      languageCode: 'en-AU',
      voices: [
        { name: 'en-AU-Standard-A', gender: 'Female', type: 'Standard' },
        { name: 'en-AU-Standard-B', gender: 'Male', type: 'Standard' },
        { name: 'en-AU-Standard-C', gender: 'Female', type: 'Standard' },
        { name: 'en-AU-Standard-D', gender: 'Male', type: 'Standard' },
        { name: 'en-AU-Wavenet-A', gender: 'Female', type: 'WaveNet' },
        { name: 'en-AU-Wavenet-B', gender: 'Male', type: 'WaveNet' },
        { name: 'en-AU-Wavenet-C', gender: 'Female', type: 'WaveNet' },
        { name: 'en-AU-Wavenet-D', gender: 'Male', type: 'WaveNet' },
      ],
    },

    // **English (India)**
    {
      languageCode: 'en-IN',
      voices: [
        { name: 'en-IN-Standard-A', gender: 'Female', type: 'Standard' },
        { name: 'en-IN-Standard-B', gender: 'Male', type: 'Standard' },
        { name: 'en-IN-Wavenet-A', gender: 'Female', type: 'WaveNet' },
        { name: 'en-IN-Wavenet-B', gender: 'Male', type: 'WaveNet' },
      ],
    },

    // **English (United States)**
    {
      languageCode: 'en-US',
      voices: [
        { name: 'en-US-Standard-A', gender: 'Female', type: 'Standard' },
        { name: 'en-US-Standard-B', gender: 'Male', type: 'Standard' },
        { name: 'en-US-Standard-C', gender: 'Female', type: 'Standard' },
        { name: 'en-US-Standard-D', gender: 'Male', type: 'Standard' },
        { name: 'en-US-Standard-E', gender: 'Female', type: 'Standard' },
        { name: 'en-US-Standard-F', gender: 'Female', type: 'Standard' },
        { name: 'en-US-Wavenet-A', gender: 'Female', type: 'WaveNet' },
        { name: 'en-US-Wavenet-B', gender: 'Male', type: 'WaveNet' },
        { name: 'en-US-Wavenet-C', gender: 'Female', type: 'WaveNet' },
        { name: 'en-US-Wavenet-D', gender: 'Male', type: 'WaveNet' },
        { name: 'en-US-Wavenet-E', gender: 'Female', type: 'WaveNet' },
        { name: 'en-US-Wavenet-F', gender: 'Female', type: 'WaveNet' },
      ],
    },

    // **Spanish (Spain)**
    {
      languageCode: 'es-ES',
      voices: [
        { name: 'es-ES-Standard-A', gender: 'Female', type: 'Standard' },
        { name: 'es-ES-Standard-B', gender: 'Male', type: 'Standard' },
        { name: 'es-ES-Wavenet-A', gender: 'Female', type: 'WaveNet' },
        { name: 'es-ES-Wavenet-B', gender: 'Male', type: 'WaveNet' },
      ],
    },

    // **Spanish (Mexico)**
    {
      languageCode: 'es-MX',
      voices: [
        { name: 'es-MX-Standard-A', gender: 'Female', type: 'Standard' },
        { name: 'es-MX-Standard-B', gender: 'Male', type: 'Standard' },
        { name: 'es-MX-Wavenet-A', gender: 'Female', type: 'WaveNet' },
        { name: 'es-MX-Wavenet-B', gender: 'Male', type: 'WaveNet' },
      ],
    },

    // **Spanish (United States)**
    {
      languageCode: 'es-US',
      voices: [
        { name: 'es-US-Standard-A', gender: 'Female', type: 'Standard' },
        { name: 'es-US-Standard-B', gender: 'Male', type: 'Standard' },
        { name: 'es-US-Wavenet-A', gender: 'Female', type: 'WaveNet' },
        { name: 'es-US-Wavenet-B', gender: 'Male', type: 'WaveNet' },
      ],
    },

    // **Korean**
    {
      languageCode: 'ko-KR',
      voices: [
        { name: 'ko-KR-Standard-A', gender: 'Female', type: 'Standard' },
        { name: 'ko-KR-Standard-B', gender: 'Male', type: 'Standard' },
        { name: 'ko-KR-Wavenet-A', gender: 'Female', type: 'WaveNet' },
        { name: 'ko-KR-Wavenet-B', gender: 'Male', type: 'WaveNet' },
      ],
    },
  ];
  
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

      {/* English Voice Settings Dropdown */}
      <div style={{ marginBottom: '20px' }}>
        <label htmlFor="english-voice-select">English Voice Settings:</label><br />
        <select
          id="english-voice-select"
          value={englishVoiceSetting}
          onChange={(e) => setEnglishVoiceSetting(e.target.value)}
          style={{ width: '300px', padding: '5px', marginTop: '5px' }}
        >
          <option value="">Select an English voice</option>
          {gcpVoiceOptions
            .filter((languageOption) => languageOption.languageCode.startsWith('en-'))
            .flatMap((languageOption) =>
              languageOption.voices.map((voice) => (
                <option key={voice.name} value={voice.name}>
                  {`${voice.name} (${voice.gender}, ${voice.type})`}
                </option>
              ))
            )}
        </select>
      </div>

      {/* Spanish Voice Settings Dropdown */}
      <div style={{ marginBottom: '20px' }}>
        <label htmlFor="spanish-voice-select">Spanish Voice Settings:</label><br />
        <select
          id="spanish-voice-select"
          value={spanishVoiceSetting}
          onChange={(e) => setSpanishVoiceSetting(e.target.value)}
          style={{ width: '300px', padding: '5px', marginTop: '5px' }}
        >
          <option value="">Select a Spanish voice</option>
          {gcpVoiceOptions
            .filter((languageOption) => languageOption.languageCode.startsWith('es-'))
            .flatMap((languageOption) =>
              languageOption.voices.map((voice) => (
                <option key={voice.name} value={voice.name}>
                  {`${voice.name} (${voice.gender}, ${voice.type})`}
                </option>
              ))
            )}
        </select>
      </div>

      {/* Korean Voice Settings Dropdown */}
      <div style={{ marginBottom: '20px' }}>
        <label htmlFor="korean-voice-select">Korean Voice Settings:</label><br />
        <select
          id="korean-voice-select"
          value={koreanVoiceSetting}
          onChange={(e) => setKoreanVoiceSetting(e.target.value)}
          style={{ width: '300px', padding: '5px', marginTop: '5px' }}
        >
          <option value="">Select a Korean voice</option>
          {gcpVoiceOptions
            .filter((languageOption) => languageOption.languageCode === 'ko-KR')
            .flatMap((languageOption) =>
              languageOption.voices.map((voice) => (
                <option key={voice.name} value={voice.name}>
                  {`${voice.name} (${voice.gender}, ${voice.type})`}
                </option>
              ))
            )}
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
