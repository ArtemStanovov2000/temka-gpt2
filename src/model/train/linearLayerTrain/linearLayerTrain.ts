// Константы
const VOCAB_SIZE = 65;
const CONTEXT_LENGTH = 64;
const EMBEDDING_DIM = 10;

// Softmax функция
const softmax = (row: number[]): number[] => {
    const max = Math.max(...row);
    const exps = row.map(x => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(x => x / sum);
};

// Прямой проход линейного слоя
export const linearForward = (input: number[][], weights: number[][]): number[][] => {
    const output: number[][] = [];
    for (let i = 0; i < CONTEXT_LENGTH; i++) {
        output[i] = new Array(VOCAB_SIZE).fill(0);
        for (let j = 0; j < VOCAB_SIZE; j++) {
            for (let k = 0; k < EMBEDDING_DIM; k++) {
                output[i][j] += input[i][k] * weights[k][j];
            }
        }
    }
    return output;
};

// Вычисление потерь
export const calculateLoss = (logits: number[][], targets: number[]): number => {
    let loss = 0;
    for (let i = 0; i < CONTEXT_LENGTH - 1; i++) {
        const probs = softmax(logits[i]);
        loss += -Math.log(probs[targets[i + 1]] + 1e-10); // Добавляем небольшое значение для стабильности
    }
    return loss / (CONTEXT_LENGTH - 1);
};

// Обратное распространение для линейного слоя
export const linearBackward = (
    input: number[][],
    gradOutput: number[][],
    weights: number[][]
): { gradWeights: number[][], gradInput: number[][] } => {
    const gradWeights: number[][] = Array.from({ length: EMBEDDING_DIM }, () => 
        new Array(VOCAB_SIZE).fill(0));
    
    const gradInput: number[][] = Array.from({ length: CONTEXT_LENGTH }, () => 
        new Array(EMBEDDING_DIM).fill(0));

    // Вычисляем градиенты для весов
    for (let i = 0; i < CONTEXT_LENGTH; i++) {
        for (let j = 0; j < VOCAB_SIZE; j++) {
            for (let k = 0; k < EMBEDDING_DIM; k++) {
                gradWeights[k][j] += input[i][k] * gradOutput[i][j];
            }
        }
    }

    // Вычисляем градиенты для входных данных
    for (let i = 0; i < CONTEXT_LENGTH; i++) {
        for (let j = 0; j < EMBEDDING_DIM; j++) {
            for (let k = 0; k < VOCAB_SIZE; k++) {
                gradInput[i][j] += weights[j][k] * gradOutput[i][k];
            }
        }
    }

    return { gradWeights, gradInput };
};

// Градиент для softmax и cross-entropy
export const softmaxCrossEntropyBackward = (
    logits: number[][],
    targets: number[]
): number[][] => {
    const gradOutput: number[][] = Array.from({ length: CONTEXT_LENGTH }, () => 
        new Array(VOCAB_SIZE).fill(0));
    
    for (let i = 0; i < CONTEXT_LENGTH - 1; i++) {
        const probs = softmax(logits[i]);
        for (let j = 0; j < VOCAB_SIZE; j++) {
            gradOutput[i][j] = probs[j] - (j === targets[i + 1] ? 1 : 0);
        }
    }
    //console.log(gradOutput, "loss")
    return gradOutput;
};

// Обновление весов
export const updateWeights = (
    weights: number[][],
    gradWeights: number[][],
    learningRate: number
): void => {
    for (let i = 0; i < weights.length; i++) {
        for (let j = 0; j < weights[i].length; j++) {
            weights[i][j] -= learningRate * gradWeights[i][j];
        }
    }
};