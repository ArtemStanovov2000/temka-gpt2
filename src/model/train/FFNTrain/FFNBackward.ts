import { matrixMultiply } from "../../FFN/FFN";
import { clipGradients } from "../../../utils/clipGradients/clipGradients";
import { clipGradientsVector } from "../../../utils/clipGradients/clipGradientsVector";

// Производная функции GELU
function geluDerivative(x: number): number {
    const x3 = x * x * x;
    return 0.89894228 * (2 * x + 0.17886 * x3);
}

// Транспонирование матрицы
function transpose(matrix: number[][]): number[][] {
    const rows = matrix.length;
    const cols = matrix[0].length;
    const result: number[][] = Array.from({ length: cols }, () => new Array(rows).fill(0));

    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            result[j][i] = matrix[i][j];
        }
    }
    return result;
}

// Поэлементное умножение матриц
function elementwiseMultiply(A: number[][], B: number[][]): number[][] {
    const rows = A.length;
    const cols = A[0].length;
    const result: number[][] = Array.from({ length: rows }, () => new Array(cols).fill(0));

    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            result[i][j] = A[i][j] * B[i][j];
        }
    }
    return result;
}

// Сумма по столбцам матрицы
function sumColumns(matrix: number[][]): number[] {
    const rows = matrix.length;
    const cols = matrix[0].length;
    const result: number[] = new Array(cols).fill(0);

    for (let j = 0; j < cols; j++) {
        for (let i = 0; i < rows; i++) {
            result[j] += matrix[i][j];
        }
    }
    return result;
}


export const FFNBackward = (
    dOutput: number[][], // Градиент от следующего слоя [64, 10]
    cache: any,
    weights1: number[][],
    weights2: number[][],
    learningRate: number = 0.01
): { dInput: number[][], dW1: number[][], db1: number[], dW2: number[][], db2: number[] } => {
    // Извлекаем промежуточные результаты из кэша
    const { input, hidden, hiddenWithBias, activated, outputBeforeBias } = cache;

    // Градиенты для параметров
    const dW2: number[][] = Array.from({ length: weights2.length }, () =>
        new Array(weights2[0].length).fill(0));
    const db2: number[] = new Array(weights2[0].length).fill(0);
    const dW1: number[][] = Array.from({ length: weights1.length }, () =>
        new Array(weights1[0].length).fill(0));
    const db1: number[] = new Array(weights1[0].length).fill(0);

    // 1. Backprop through output bias
    for (let i = 0; i < dOutput.length; i++) {
        for (let j = 0; j < dOutput[0].length; j++) {
            db2[j] += dOutput[i][j];
        }
    }

    // 2. Backprop through output linear transformation
    const dOutputBeforeBias = dOutput; // Так как bias - это просто добавление
    const dActivated = matrixMultiply(dOutputBeforeBias, transpose(weights2));

    // Градиент для weights2
    const activatedT = transpose(activated);
    const dW2Add = matrixMultiply(activatedT, dOutputBeforeBias);

    for (let i = 0; i < dW2.length; i++) {
        for (let j = 0; j < dW2[0].length; j++) {
            dW2[i][j] = dW2Add[i][j];
        }
    }

    // 3. Backprop through GELU activation
    const dHiddenWithBias = elementwiseMultiply(
        dActivated,
        hiddenWithBias.map((row: number[]) => row.map(geluDerivative))
    );

    // 4. Backprop through hidden bias
    for (let i = 0; i < dHiddenWithBias.length; i++) {
        for (let j = 0; j < dHiddenWithBias[0].length; j++) {
            db1[j] += dHiddenWithBias[i][j];
        }
    }

    // 5. Backprop through hidden linear transformation
    const dHidden = dHiddenWithBias; // Так как bias - это просто добавление
    const dInput = matrixMultiply(dHidden, transpose(weights1));

    // Градиент для weights1
    const inputT = transpose(input);
    const dW1Add = matrixMultiply(inputT, dHidden);

    for (let i = 0; i < dW1.length; i++) {
        for (let j = 0; j < dW1[0].length; j++) {
            dW1[i][j] = dW1Add[i][j];
        }
    }

    return { dInput, dW1: clipGradients(dW1, 2), db1: clipGradientsVector(db1, 2), dW2: clipGradients(dW2, 2), db2: clipGradientsVector(db2, 2) };
};

export const updateFFNWeights = (
    weights1: number[][],
    weights2: number[][],
    biasHidden: number[],
    biasOutput: number[],
    dW1: number[][],
    dW2: number[][],
    db1: number[],
    db2: number[],
    learningRate: number
): void => {
    // Обновляем weights1
    for (let i = 0; i < weights1.length; i++) {
        for (let j = 0; j < weights1[0].length; j++) {
            weights1[i][j] -= learningRate * dW1[i][j];
        }
    }

    // Обновляем weights2
    for (let i = 0; i < weights2.length; i++) {
        for (let j = 0; j < weights2[0].length; j++) {
            weights2[i][j] -= learningRate * dW2[i][j];
        }
    }

    // Обновляем biasHidden
    for (let j = 0; j < biasHidden.length; j++) {
        biasHidden[j] -= learningRate * db1[j];
    }

    // Обновляем biasOutput
    for (let j = 0; j < biasOutput.length; j++) {
        biasOutput[j] -= learningRate * db2[j];
    }
};