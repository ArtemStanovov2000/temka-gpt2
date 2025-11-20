import { matrixMultiply } from "../../FFN/FFN";
import { matrixTranspose } from "../../attention/attention";
import { matrixAdd } from "../../attention/attention";
import { clipGradients } from "../../../utils/clipGradients/clipGradients";

export const attentionBackward = (
    dOutput: number[][],
    cache: any,
    learningRate: number = 0.01
) => {
    // 1. Backprop через второй слой нормализации
    const { dInput: dNorm2, dGamma: dGamma2, dBeta: dBeta2 } = layerNormBackward(dOutput, cache.norm2Cache);

    // 2. Backprop через residual connection
    const dResidual = dNorm2;
    const dProjOutput = dNorm2;

    // 3. Backprop через выходную проекцию
    const dAttentionOutput = matrixMultiply(dProjOutput, matrixTranspose(cache.W_O));
    const dW_O = clipGradients(matrixMultiply(matrixTranspose(cache.attentionOutput), dProjOutput), 2);

    // 4. Backprop через взвешенную сумму
    const dWeights = matrixMultiply(dAttentionOutput, matrixTranspose(cache.V));
    const dV = matrixMultiply(matrixTranspose(cache.weights), dAttentionOutput);

    // 5. Backprop через softmax
    const dScoresAfterMask = softmaxBackward(dWeights, cache.softmaxCache);

    // 6. Backprop через mask
    const dScoresAfterScale = dScoresAfterMask;

    // 7. Backprop через масштабирование
    const dScoresBeforeScale = dScoresAfterScale.map(row =>
        row.map(val => val / Math.sqrt(cache.Q[0].length))
    );

    // 8. Backprop через матрицу внимания
    const dQ = matrixMultiply(dScoresBeforeScale, cache.K);
    const dK = matrixMultiply(matrixTranspose(dScoresBeforeScale), cache.Q);

    // 9. Backprop через Q, K, V проекции
    const dNorm1 = matrixAdd(
        matrixAdd(
            matrixMultiply(dQ, matrixTranspose(cache.W_Q)),
            matrixMultiply(dK, matrixTranspose(cache.W_K))
        ),
        matrixMultiply(dV, matrixTranspose(cache.W_V))
    );

    const dW_Q = clipGradients(matrixMultiply(matrixTranspose(cache.norm1), dQ), 2);
    const dW_K = clipGradients(matrixMultiply(matrixTranspose(cache.norm1), dK), 2);
    const dW_V = clipGradients(matrixMultiply(matrixTranspose(cache.norm1), dV), 2);

    // 10. Backprop через первый слой нормализации
    const { dInput: dFirstNorm, dGamma: dGamma1, dBeta: dBeta1 } = layerNormBackward(dNorm1, cache.norm1Cache);

    // 11. Combine gradients from both paths
    const dInput = matrixAdd(dFirstNorm, dResidual);

    return {
        dInput,
        dGamma1,
        dBeta1,
        dGamma2,
        dBeta2,
        dW_Q,
        dW_K,
        dW_V,
        dW_O
    };
};

const layerNormBackward = (dOutput: number[][], cache: any) => {
    const { input, gamma, mean, variance, stdDev } = cache;
    const dGamma = new Array(gamma.length).fill(0);
    const dBeta = new Array(gamma.length).fill(0);
    const dInput: number[][] = Array.from({ length: input.length }, () =>
        new Array(input[0].length).fill(0)
    );

    for (let i = 0; i < input.length; i++) {
        const xMinusMean = input[i].map((x: number) => x - mean[i]);
        const invStdDev = 1 / stdDev[i];

        // Calculate dGamma and dBeta
        for (let j = 0; j < input[0].length; j++) {
            dGamma[j] += dOutput[i][j] * xMinusMean[j] * invStdDev;
            dBeta[j] += dOutput[i][j];
        }

        // Calculate dInput
        const dxHat = dOutput[i].map((val, j) => val * gamma[j]);
        const dVar = dxHat.reduce((sum, val, j) =>
            sum + val * xMinusMean[j] * -0.5 * Math.pow(variance[i] + 1e-8, -1.5), 0
        );

        const dMean = dxHat.reduce((sum, val) =>
            sum + val * -invStdDev, 0
        ) + dVar * -2 * xMinusMean.reduce((sum: number, val: number) => sum + val, 0) / input[0].length;

        for (let j = 0; j < input[0].length; j++) {
            dInput[i][j] = dxHat[j] * invStdDev +
                dVar * 2 * xMinusMean[j] / input[0].length +
                dMean / input[0].length;
        }
    }

    return { dInput: clipGradients(dInput, 2), dGamma, dBeta };
};

const softmaxBackward = (dOutput: number[][], cache: any) => {
    const { input, maxVal, expSum } = cache;
    const dInput: number[][] = Array.from({ length: input.length }, () =>
        new Array(input[0].length).fill(0)
    );

    for (let i = 0; i < input.length; i++) {
        // Восстанавливаем exp values из кэша
        const expRow: number[] = [];
        for (let j = 0; j < input[0].length; j++) {
            const expVal = Math.exp(input[i][j] - maxVal[i]);
            expRow.push(expVal);
        }

        // Вычисляем сумму произведений dOutput и output
        let sum = 0;
        for (let j = 0; j < input[0].length; j++) {
            const outputVal = expRow[j] / expSum[i];
            sum += dOutput[i][j] * outputVal;
        }

        // Вычисляем градиенты
        for (let j = 0; j < input[0].length; j++) {
            const outputVal = expRow[j] / expSum[i];
            dInput[i][j] = outputVal * (dOutput[i][j] - sum);
        }
    }

    return clipGradients(dInput, 2);
};