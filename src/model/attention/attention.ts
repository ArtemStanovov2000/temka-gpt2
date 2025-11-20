function matrixMultiply(A: number[][], B: number[][]) {
    const rowsA = A.length;
    const colsA = A[0].length;
    const rowsB = B.length;
    const colsB = B[0].length;
    
    if (colsA !== rowsB) {
        throw new Error("Несовместимые размеры матриц");
    }
    
    const result: number[][] = [];
    for (let i = 0; i < rowsA; i++) {
        result[i] = [];
        for (let j = 0; j < colsB; j++) {
            result[i][j] = 0;
            for (let k = 0; k < colsA; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}

export const matrixTranspose = (matrix: number[][]) => {
    const rows = matrix.length;
    const cols = matrix[0].length;
    const result: number[][] = [];
    
    for (let j = 0; j < cols; j++) {
        result[j] = [];
        for (let i = 0; i < rows; i++) {
            result[j][i] = matrix[i][j];
        }
    }
    return result;
}

export const matrixAdd = (A: number[][], B: number[][]) => {
    const rows = A.length;
    const cols = A[0].length;
    const result: number[][] = [];
    
    for (let i = 0; i < rows; i++) {
        result[i] = [];
        for (let j = 0; j < cols; j++) {
            result[i][j] = A[i][j] + B[i][j];
        }
    }
    return result;
}

const layerNorm = (input: number[][], gamma: number[], beta: number[]) => {
    const epsilon = 1e-8;
    const output = [];
    
    for (let i = 0; i < input.length; i++) {
        const row = input[i];
        const mean = row.reduce((sum: number, val: number) => sum + val, 0) / row.length;
        const variance = row.reduce((sum: number, val: number) => sum + Math.pow(val - mean, 2), 0) / row.length;
        const stdDev = Math.sqrt(variance + epsilon);
        
        const normalizedRow = [];
        for (let j = 0; j < row.length; j++) {
            normalizedRow[j] = (row[j] - mean) / stdDev;
            normalizedRow[j] = normalizedRow[j] * gamma[j] + beta[j];
        }
        output.push(normalizedRow);
    }
    return output;
}

const softmax = (matrix: number[][]) => {
    const output = [];
    
    for (let i = 0; i < matrix.length; i++) {
        const row = matrix[i];
        const maxVal = Math.max(...row);
        let expSum = 0;
        const expRow = [];
        
        for (let j = 0; j < row.length; j++) {
            const expVal = Math.exp(row[j] - maxVal);
            expRow.push(expVal);
            expSum += expVal;
        }
        
        const softmaxRow = [];
        for (let j = 0; j < expRow.length; j++) {
            softmaxRow.push(expRow[j] / expSum);
        }
        output.push(softmaxRow);
    }
    return output;
}

const createCausalMask = (sequenceLength: number) => {
    const mask: number[][] = [];
    for (let i = 0; i < sequenceLength; i++) {
        mask[i] = [];
        for (let j = 0; j < sequenceLength; j++) {
            // Разрешаем внимание только к прошлым и текущему токену
            mask[i][j] = (j > i) ? -1e9 : 0;
        }
    }
    return mask;
}

export const attention = (embeddings: number[][], gamma1: number[], beta1: number[], gamma2: number[], beta2: number[], W_Q: number[][], W_K: number[][], W_V: number[][], W_O: number[][]) => {
    // 1. Первый слой нормализации
    const norm1 = layerNorm(embeddings, gamma1, beta1);

    // 2. Вычисление Q, K, V
    const Q = matrixMultiply(norm1, W_Q);
    const K = matrixMultiply(norm1, W_K);
    const V = matrixMultiply(norm1, W_V);

    // 3. Вычисление матрицы внимания
    const K_T = matrixTranspose(K);
    let scores = matrixMultiply(Q, K_T);

    // 4. Масштабирование
    const d_k = Q[0].length;
    for (let i = 0; i < scores.length; i++) {
        for (let j = 0; j < scores[0].length; j++) {
            scores[i][j] /= Math.sqrt(d_k);
        }
    }

    // 5. Применение causal mask
    const mask = createCausalMask(scores.length);
    for (let i = 0; i < scores.length; i++) {
        for (let j = 0; j < scores[0].length; j++) {
            scores[i][j] += mask[i][j];
        }
    }

    // 6. Softmax
    const weights = softmax(scores);

    // 7. Взвешенная сумма значений
    const attentionOutput = matrixMultiply(weights, V);

    // 8. Выходная проекция
    const projOutput = matrixMultiply(attentionOutput, W_O);

    // 9. Residual connection
    const residual = matrixAdd(embeddings, projOutput);

    // 10. Второй слой нормализации
    const output = layerNorm(residual, gamma2, beta2);

    return output;
}