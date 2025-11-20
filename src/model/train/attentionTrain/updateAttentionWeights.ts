export const updateAttentionWeights = (
    W_Q: number[][],
    W_K: number[][],
    W_V: number[][],
    W_O: number[][],
    gamma1: number[],
    beta1: number[],
    gamma2: number[],
    beta2: number[],
    dW_Q: number[][],
    dW_K: number[][],
    dW_V: number[][],
    dW_O: number[][],
    dGamma1: number[],
    dBeta1: number[],
    dGamma2: number[],
    dBeta2: number[],
    learningRate: number
): void => {
    // Обновляем веса attention
    updateMatrix(W_Q, dW_Q, learningRate);
    updateMatrix(W_K, dW_K, learningRate);
    updateMatrix(W_V, dW_V, learningRate);
    updateMatrix(W_O, dW_O, learningRate);
    
    // Обновляем параметры нормализации
    updateVector(gamma1, dGamma1, learningRate);
    updateVector(beta1, dBeta1, learningRate);
    updateVector(gamma2, dGamma2, learningRate);
    updateVector(beta2, dBeta2, learningRate);
};

// Вспомогательные функции для обновления матриц и векторов
const updateMatrix = (matrix: number[][], grad: number[][], learningRate: number): void => {
    for (let i = 0; i < matrix.length; i++) {
        for (let j = 0; j < matrix[0].length; j++) {
            matrix[i][j] -= learningRate * grad[i][j];
        }
    }
};

const updateVector = (vector: number[], grad: number[], learningRate: number): void => {
    for (let i = 0; i < vector.length; i++) {
        vector[i] -= learningRate * grad[i];
    }
};