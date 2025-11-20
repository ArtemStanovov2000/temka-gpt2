import { vocab } from "../../utils/config/vocab";

// Константа температуры (можно вынести в конфиг)
export const TEMPERATURE = 0.2; // Значение от 0.1 до 1.5 для контроля случайности

export const softmax = (row: number[], temperature: number = 1.0) => {
    // Применяем температуру к логитам
    const scaled = row.map(x => x / temperature);
    const max = Math.max(...scaled);
    const exps = scaled.map(x => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(x => x / sum);
}

export const calculateLogits = (inputLayer: number[], weights: number[][]) => {
    const numTokens = weights[0].length;
    const activationLayer: number[] = new Array(numTokens).fill(0);

    for (let tokenIdx = 0; tokenIdx < numTokens; tokenIdx++) {
        let weightedSum = 0;
        for (let embedIdx = 0; embedIdx < inputLayer.length; embedIdx++) {
            weightedSum += weights[embedIdx][tokenIdx] * inputLayer[embedIdx];
        }
        activationLayer[tokenIdx] = weightedSum;
    }

    return activationLayer;
}

export const predictNextToken = (
    ffnOutput: number[][], 
    contextLength: number, 
    linearWeights: number[][], 
    temperature: number = TEMPERATURE // Добавляем параметр температуры
) => {
    // 1. Вычисляем логиты и вероятности для всех позиций
    const probabilities: number[][] = [];
    for (let i = 0; i < contextLength; i++) {
        const logits = calculateLogits(ffnOutput[i], linearWeights);
        // Применяем температуру при вычислении softmax
        probabilities.push(softmax(logits, temperature));
    }

    // 2. Берем вероятности для последней релевантной позиции
    const lastPositionProbs = probabilities[contextLength - 1];

    // 3. Сэмплируем токен с учетом температуры
    return sampleFromDistribution(lastPositionProbs);
}

// Новая функция для сэмплирования из распределения
export const sampleFromDistribution = (probs: number[]): string => {
    const random = Math.random();
    let cumulativeProb = 0;
    
    for (let i = 0; i < probs.length; i++) {
        cumulativeProb += probs[i];
        if (random <= cumulativeProb) {
            return vocab[i];
        }
    }
    
    // Fallback: возвращаем токен с максимальной вероятностью
    const maxIndex = probs.indexOf(Math.max(...probs));
    return vocab[maxIndex];
}

// Дополнительная функция для получения топ-N токенов (опционально)
export const getTopTokens = (
    probs: number[], 
    topK: number = 5
): Array<{ token: string, probability: number }> => {
    const indexedProbs = probs.map((prob, index) => ({ index, prob }));
    indexedProbs.sort((a, b) => b.prob - a.prob);
    
    return indexedProbs.slice(0, topK).map(item => ({
        token: vocab[item.index],
        probability: item.prob
    }));
}