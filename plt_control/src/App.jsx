import React, { useState } from 'react';

function App() {
  const [baseLanguage, setBaseLanguage] = useState('');
  // const [targetLanguage, setTargetLanguage] = useState('');
  const [gender, setGender] = useState('');

  const languageMapping = {
    english: 'en-US',
    spanish: 'es-US',
    korean: 'ko-KR',
  };

  const handleConfirm = async () => {
    // if (!baseLanguage || !targetLanguage || !gender) {
    //   alert('Please fill in all fields.');
    //   return;
    // }
    const settings = {
      baseLanguage: languageMapping[baseLanguage],
      // targetLanguage: languageMapping[targetLanguage],
      gender,
    };

    try {
      const response = await fetch('http://128.197.180.248:5000/set_settings', {
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
        {/* <div style={{ marginBottom: '20px' }}>
          <label htmlFor="target-language" style={{ display: 'block', marginBottom: '8px' }}>Translating Language:</label>
          <select
            id="target-language"
            value={targetLanguage}
            onChange={(e) => setTargetLanguage(e.target.value)}
            style={{
              width: '100%',
              padding: '10px',
              borderRadius: '4px',
              border: '1px solid #F77F00',
              backgroundColor: '#000000',
              color: '#FFFFFF'
            }}
          >
            <option value="">Select Translating Language</option>
            <option value="english">English</option>
            <option value="spanish">Spanish</option>
            <option value="korean">Korean</option>
          </select>
        </div> */}
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

// import React, { useState } from 'react';

// function App() {
//   // State for base language
//   const [baseLanguage, setBaseLanguage] = useState('');

//   // State for gender selection
//   const [gender, setGender] = useState('');

//   const baseLanguageMapping = {
//     english: 'en-US',
//     spanish: 'es-US',
//     korean: 'ko-KR',
//   };

//   const handleConfirm = () => {
//     if (!baseLanguage) {
//       alert('Please select a base language.');
//       return;
//     }
//     if (!gender) {
//       alert('Please select a gender.');
//       return;
//     }

//     const settings = {
//       baseLanguage: baseLanguageMapping[baseLanguage],
//       gender: gender,
//     };

//     // Send settings to the backend
//     // fetch('http://localhost:5000/set_settings', {
//     fetch ('http://128.197.180.248:5000/set_settings', {
//       method: 'POST',
//       headers: {
//         'Content-Type': 'application/json',
//       },
//       body: JSON.stringify(settings),
//     })
//       .then((response) => response.json())
//       .then((data) => {
//         console.log('Settings updated:', data);
//       })
//       .catch((error) => {
//         console.error('Error updating settings:', error);
//       });
//   };

//   return (
//     <div style={{ padding: '20px', fontFamily: 'Arial' }}>
//       <h1>Settings</h1>

//       {/* Base Language Selection Dropdown */}
//       <div style={{ marginBottom: '20px' }}>
//         <label htmlFor="language-select">Base Language:</label><br />
//         <select
//           id="language-select"
//           value={baseLanguage}
//           onChange={(e) => setBaseLanguage(e.target.value)}
//           style={{ width: '200px', padding: '5px', marginTop: '5px' }}
//         >
//           <option value="">Select Base Language</option>
//           <option value="english">English</option>
//           <option value="spanish">Spanish</option>
//           <option value="korean">Korean</option>
//         </select>
//       </div>

//       {/* Gender Selection Dropdown */}
//       <div style={{ marginBottom: '20px' }}>
//         <label htmlFor="gender-select">Select Gender:</label><br />
//         <select
//           id="gender-select"
//           value={gender}
//           onChange={(e) => setGender(e.target.value)}
//           style={{ width: '200px', padding: '5px', marginTop: '5px' }}
//         >
//           <option value="">Select Gender</option>
//           <option value="FEMALE">Female</option>
//           <option value="MALE">Male</option>
//         </select>
//       </div>

//       {/* Confirm Button */}
//       <button
//         onClick={handleConfirm}
//         style={{
//           padding: '10px 20px',
//           backgroundColor: '#6200ee',
//           color: '#fff',
//           border: 'none',
//           cursor: 'pointer',
//         }}
//       >
//         Confirm
//       </button>
//     </div>
//   );
// }

// export default App;
