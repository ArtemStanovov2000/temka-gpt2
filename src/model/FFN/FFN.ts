export const matrixMultiply = (A: number[][], B: number[][]) => {
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

// Вспомогательная функция: добавление bias к матрице
export const addBias = (matrix: number[][], bias: number[]) => {
    const result: number[][] = [];
    for (let i = 0; i < matrix.length; i++) {
        result[i] = [];
        for (let j = 0; j < matrix[i].length; j++) {
            result[i][j] = matrix[i][j] + bias[j];
        }
    }
    return result;
}

export const gelu = (x: number) => {
    return 0.5 * x * (1.79788456 * (x + 0.044715 * Math.pow(x, 3)))
}

// Вспомогательная функция: применение активации к матрице
export const applyActivation = (matrix: number[][], activationFn: (x: number) => number) => {
    return matrix.map(row => row.map(activationFn));
}

export const FFN = (input: number[][], weights1: number[][], weights2: number[][], biasHidden: number[], biasOutput: number[]) => {
    // Шаг 1: Первое линейное преобразование (расширение размерности)
    const hidden = matrixMultiply(input, weights1);
    
    // Шаг 2: Добавление смещения
    const hiddenWithBias = addBias(hidden, biasHidden);
    
    // Шаг 3: Применение активации GELU
    const activated = applyActivation(hiddenWithBias, gelu);
    
    // Шаг 4: Второе линейное преобразование (сжатие размерности)
    const output = matrixMultiply(activated, weights2);
    
    // Шаг 5: Добавление выходного смещения
    const outputWithBias = addBias(output, biasOutput);
    
    return outputWithBias;
}