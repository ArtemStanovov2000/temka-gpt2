import { matrixMultiply } from "../../FFN/FFN";

const matrixTranspose = (matrix: number[][]) => {
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

const matrixAdd = (A: number[][], B: number[][]) => {
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

export const attentionForward = (
    embeddings: number[][],
    gamma1: number[],
    beta1: number[],
    gamma2: number[],
    beta2: number[],
    W_Q: number[][],
    W_K: number[][],
    W_V: number[][],
    W_O: number[][]
): { output: number[][], cache: any } => {
    const cache: any = {};
    
    // 1. Первый слой нормализации
    const norm1 = layerNormWithCache(embeddings, gamma1, beta1);
    cache.norm1 = norm1.output;
    cache.norm1Cache = norm1.cache;
    
    // 2. Вычисление Q, K, V
    const Q = matrixMultiply(norm1.output, W_Q);
    const K = matrixMultiply(norm1.output, W_K);
    const V = matrixMultiply(norm1.output, W_V);
    cache.Q = Q;
    cache.K = K;
    cache.V = V;
    cache.W_Q = W_Q;
    cache.W_K = W_K;
    cache.W_V = W_V;
    
    // 3. Вычисление матрицы внимания
    const K_T = matrixTranspose(K);
    let scores = matrixMultiply(Q, K_T);
    cache.scoresBeforeScale = [...scores.map(row => [...row])]; // Сохраняем копию
    
    // 4. Масштабирование
    const d_k = Q[0].length;
    for (let i = 0; i < scores.length; i++) {
        for (let j = 0; j < scores[0].length; j++) {
            scores[i][j] /= Math.sqrt(d_k);
        }
    }
    cache.scoresAfterScale = [...scores.map(row => [...row])]; // Сохраняем копию
    
    // 5. Применение causal mask
    const mask = createCausalMask(scores.length);
    cache.mask = mask;
    for (let i = 0; i < scores.length; i++) {
        for (let j = 0; j < scores[0].length; j++) {
            scores[i][j] += mask[i][j];
        }
    }
    cache.scoresAfterMask = [...scores.map(row => [...row])]; // Сохраняем копию
    
    // 6. Softmax
    const weights = softmaxWithCache(scores);
    cache.weights = weights.output;
    cache.softmaxCache = weights.cache;
    
    // 7. Взвешенная сумма значений
    const attentionOutput = matrixMultiply(weights.output, V);
    cache.attentionOutput = attentionOutput;
    
    // 8. Выходная проекция
    const projOutput = matrixMultiply(attentionOutput, W_O);
    cache.projOutput = projOutput;
    cache.W_O = W_O;
    
    // 9. Residual connection
    const residual = matrixAdd(embeddings, projOutput);
    cache.residualInput = embeddings; // Сохраняем исходные embeddings
    
    // 10. Второй слой нормализации
    const norm2 = layerNormWithCache(residual, gamma2, beta2);
    cache.norm2 = norm2.output;
    cache.norm2Cache = norm2.cache;
    
    return { output: norm2.output, cache };
};

// Модифицированная функция layerNorm с сохранением промежуточных результатов
const layerNormWithCache = (input: number[][], gamma: number[], beta: number[]) => {
    const epsilon = 1e-8;
    const output = [];
    const cache: any = {
        input: [...input.map(row => [...row])],
        gamma: [...gamma],
        beta: [...beta],
        mean: [],
        variance: [],
        stdDev: []
    };
    
    for (let i = 0; i < input.length; i++) {
        const row = input[i];
        const mean = row.reduce((sum: number, val: number) => sum + val, 0) / row.length;
        const variance = row.reduce((sum: number, val: number) => sum + Math.pow(val - mean, 2), 0) / row.length;
        const stdDev = Math.sqrt(variance + epsilon);
        
        cache.mean[i] = mean;
        cache.variance[i] = variance;
        cache.stdDev[i] = stdDev;
        
        const normalizedRow = [];
        for (let j = 0; j < row.length; j++) {
            normalizedRow[j] = (row[j] - mean) / stdDev;
            normalizedRow[j] = normalizedRow[j] * gamma[j] + beta[j];
        }
        output.push(normalizedRow);
    }
    
    return { output, cache };
};

// Модифицированная функция softmax с сохранением промежуточных результатов
const softmaxWithCache = (matrix: number[][]) => {
    const output = [];
    const cache: any = {
        input: [...matrix.map(row => [...row])],
        maxVal: [],
        expSum: []
    };
    
    for (let i = 0; i < matrix.length; i++) {
        const row = matrix[i];
        const maxVal = Math.max(...row);
        let expSum = 0;
        const expRow = [];
        
        cache.maxVal[i] = maxVal;
        
        for (let j = 0; j < row.length; j++) {
            const expVal = Math.exp(row[j] - maxVal);
            expRow.push(expVal);
            expSum += expVal;
        }
        
        cache.expSum[i] = expSum;
        
        const softmaxRow = [];
        for (let j = 0; j < expRow.length; j++) {
            softmaxRow.push(expRow[j] / expSum);
        }
        output.push(softmaxRow);
    }
    
    return { output, cache };
};

// Функция для создания causal mask
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
};