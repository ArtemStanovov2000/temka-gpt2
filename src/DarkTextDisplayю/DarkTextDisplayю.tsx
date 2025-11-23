import React, { useState } from 'react';
import { stylesDarkTextDisplay } from './style';
import { createNewToken } from '../createNewToken';

const DarkTextDisplay: React.FC = () => {
  const [inputText, setInputText] = useState('Привет');
  const [displayText, setDisplayText] = useState('Привет');


  const handleStart = () => {
    setDisplayText(inputText);
    console.log(createNewToken(displayText))
  };

  return (
    <div style={stylesDarkTextDisplay.container}>
      <div style={stylesDarkTextDisplay.header}>
        <div style={stylesDarkTextDisplay.inputRow}>
          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Введите текст..."
            style={stylesDarkTextDisplay.textField}
          />
          <button
            onClick={handleStart}
            style={stylesDarkTextDisplay.button}
          >
            Старт
          </button>
        </div>
      </div>

      {displayText && (
        <div style={stylesDarkTextDisplay.displayArea}>
          {displayText}
        </div>
      )}
    </div>
  );
};

export default DarkTextDisplay;