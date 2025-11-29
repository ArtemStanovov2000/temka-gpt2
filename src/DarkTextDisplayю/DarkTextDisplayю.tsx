import React, { useState } from 'react';
import { stylesDarkTextDisplay } from './style';
import { createNewToken } from '../createNewToken';
import { train } from '../train';
import { tokensExample } from '../tokensExample';


const gradientConfig = {
  clipNorm: 1.2,
  clipValue: { min: -2.0, max: 2.0 },
  monitorFrequency: 100
};

for (let i = 0; i < 15000; i++) {
  console.log(train(tokensExample, 0.003, gradientConfig))
}

const DarkTextDisplay: React.FC = () => {
  const [inputText, setInputText] = useState('Привет');
  const [displayText, setDisplayText] = useState('Привет');
  const [isProcessing, setIsProcessing] = useState(false);

  const handleStart = async () => {
    if (isProcessing) return;

    setIsProcessing(true);
    let currentString = inputText;
    setDisplayText(currentString);

    while (currentString.length < 64) {
      try {
        // Ждем выполнения createNewToken (на случай если она асинхронная)
        const newToken = await Promise.resolve(createNewToken(currentString));
        currentString += newToken;
        setDisplayText(currentString);

        // Небольшая задержка для визуализации процесса
        await new Promise(resolve => setTimeout(resolve, 100));
      } catch (error) {
        console.error('Error in createNewToken:', error);
        break;
      }
    }

    setIsProcessing(false);
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
            disabled={isProcessing}
            style={{
              ...stylesDarkTextDisplay.button,
              ...(isProcessing ? { opacity: 0.6 } : {})
            }}
          >
            {isProcessing ? 'Обработка...' : 'Старт'}
          </button>
        </div>
      </div>

      {displayText && (
        <div style={stylesDarkTextDisplay.displayArea}>
          {displayText}
          {isProcessing && <div>Длина: {displayText.length}/64</div>}
        </div>
      )}
    </div>
  );
};

export default DarkTextDisplay;