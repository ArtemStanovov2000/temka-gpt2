import { vocab } from "./vocab";

import { tokenEmbedding } from "./matrix/tokenEmbedding";
import { positionEmbedding } from "./matrix/positionEmbedding";

import { Wk1 } from "./matrix/layer1/Wk1";
import { Wo1 } from "./matrix/layer1/Wo1";
import { Wq1 } from "./matrix/layer1/Wq1";
import { Wv1 } from "./matrix/layer1/Wv1";
import { FFN_W11 } from "./matrix/layer1/FFN_W11";
import { FFN_W21 } from "./matrix/layer1/FFN_W21";
import { betaAfterAttention_1 } from "./matrix/layer1/betaAfterAttention_1";
import { gammaAfterAttention_1 } from "./matrix/layer1/gammaAfterAttention_1";
import { betaAfterFFN_1 } from "./matrix/layer1/betaAfterFFN_1";
import { gammaAfterFFN_1 } from "./matrix/layer1/gammaAfterFFN_1";

import { Wk2 } from "./matrix/layer2/Wk2";
import { Wo2 } from "./matrix/layer2/Wo2";
import { Wq2 } from "./matrix/layer2/Wq2";
import { Wv2 } from "./matrix/layer2/Wv2";
import { FFN_W12 } from "./matrix/layer2/FFN_W12";
import { FFN_W22 } from "./matrix/layer2/FFN_W22";
import { betaAfterAttention_2 } from "./matrix/layer2/betaAfterAttention_2";
import { gammaAfterAttention_2 } from "./matrix/layer2/gammaAfterAttention_2";
import { betaAfterFFN_2 } from "./matrix/layer2/betaAfterFFN_2";
import { gammaAfterFFN_2 } from "./matrix/layer2/gammaAfterFFN_2";

import { linearLayer } from "./matrix/linearLayer";

const TEMPERATURE = 0.8

const tokenize = (text: string, vocab: { [key: number]: string }): number[] => {
    const charToIdx: { [key: string]: number } = {};

    // Создаем обратный словарь
    for (const [idx, char] of Object.entries(vocab)) {
        charToIdx[char] = parseInt(idx);
    }

    const tokens: number[] = [];
    for (const char of text) {
        tokens.push(charToIdx[char] !== undefined ? charToIdx[char] : charToIdx['[UNK]']);
    }
    return tokens;
}

const detokenize = (tokens: number[], vocab: { [key: number]: string }): string => {
    const chars: string[] = [];
    for (const token of tokens) {
        chars.push(vocab[token] !== undefined ? vocab[token] : '[UNK]');
    }
    return chars.join('');
}

const createEmbeddingsSequence = (tokens: number[]): number[][] => {
    return tokens.map(element => tokenEmbedding[element])
}

const createPozSequence = (embeddings: number[][], pozitionMatrix: number[][]): number[][] => {
    const paddingLength = pozitionMatrix.length - embeddings.length
    const paddingArray = []
    for (let i = 0; i < paddingLength; i++) {
        const paddingInstance = []
        for (let j = 0; j < embeddings[0].length; j++) {
            paddingInstance.push(0)
        }
        paddingArray.push(paddingInstance)
    }

    return embeddings.map((tokenVector, i) =>
        tokenVector.map((value, j) => value + pozitionMatrix[i][j])
    ).concat(paddingArray);
}

const matrixMultiply = (a: number[][], b: number[][]): number[][] => {
    const rowsA = a.length;
    const colsA = a[0].length;
    const rowsB = b.length;
    const colsB = b[0].length;

    if (colsA !== rowsB) {
        throw new Error("Несовместимые размеры матриц для умножения");
    }

    const result: number[][] = [];
    for (let i = 0; i < rowsA; i++) {
        result[i] = [];
        for (let j = 0; j < colsB; j++) {
            let sum = 0;
            for (let k = 0; k < colsA; k++) {
                sum += a[i][k] * b[k][j];
            }
            result[i][j] = sum;
        }
    }
    return result;
};

const matrixTranspose = (matrix: number[][]): number[][] => {
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
};

const gelu = (x: number): number => {
    return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))));
};

const matrixAdd = (a: number[][], b: number[][]): number[][] => {
    return a.map((row, i) => row.map((val, j) => val + b[i][j]));
};

const matrixScalarMultiply = (matrix: number[][], scalar: number): number[][] => {
    return matrix.map(row => row.map(val => val * scalar));
};

const softmax = (matrix: number[][]): number[][] => {
    return matrix.map(row => {
        const maxVal = Math.max(...row);
        const expRow = row.map(val => Math.exp(val - maxVal));
        const sumExp = expRow.reduce((sum, val) => sum + val, 0);
        return expRow.map(val => val / sumExp);
    });
};

const layerNorm = (
    matrix: number[][],
    gamma: number[],
    beta: number[]
): number[][] => {
    const epsilon = 1e-8;
    return matrix.map(row => {
        const mean = row.reduce((sum, val) => sum + val, 0) / row.length;
        const variance = row.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / row.length;

        return row.map((val, j) => {
            const normalized = (val - mean) / Math.sqrt(variance + epsilon);
            return normalized * gamma[j] + beta[j];
        });
    });
};

const attentionLayer = (
    input: number[][],
    Wq: number[][],
    Wk: number[][],
    Wv: number[][],
    Wo: number[][]
): number[][] => {
    const Q = matrixMultiply(input, Wq);
    const K = matrixMultiply(input, Wk);
    const V = matrixMultiply(input, Wv);

    const Kt = matrixTranspose(K);
    const attentionScores = matrixMultiply(Q, Kt);

    const dk = Q[0].length;
    const scaledAttention = matrixScalarMultiply(attentionScores, 1 / Math.sqrt(dk));

    // Применяем causal mask
    for (let i = 0; i < scaledAttention.length; i++) {
        for (let j = i + 1; j < scaledAttention[i].length; j++) {
            scaledAttention[i][j] = -Infinity;
        }
    }
    const attentionWeights = softmax(scaledAttention);  // [64, 64]
    const weightedValues = matrixMultiply(attentionWeights, V);  // [64, 10]
    const output = matrixMultiply(weightedValues, Wo);  // [64, 10]

    return output;
};



const transformerBlock = (
    input: number[][],
    layerParams: {
        Wq: number[][];
        Wk: number[][];
        Wv: number[][];
        Wo: number[][];
        FFN_W1: number[][];
        FFN_W2: number[][];
        gammaAfterAttention: number[];
        betaAfterAttention: number[];
        gammaAfterFFN: number[];
        betaAfterFFN: number[];
    }
): number[][] => {
    // 1. Self-Attention с residual connection и layer norm
    const attentionOutput = attentionLayer(input, layerParams.Wq, layerParams.Wk, layerParams.Wv, layerParams.Wo);

    // Residual connection + layer norm
    const attentionResidual = matrixAdd(input, attentionOutput);
    const normedAttention = layerNorm(attentionResidual, layerParams.gammaAfterAttention, layerParams.betaAfterAttention);

    // 2. Feed-Forward Network
    const ffnHidden = matrixMultiply(normedAttention, layerParams.FFN_W1);  // [64, 40]

    // Применяем GELU активацию
    const activatedHidden = ffnHidden.map(row => row.map(gelu));  // [64, 40]

    const ffnOutput = matrixMultiply(activatedHidden, layerParams.FFN_W2);  // [64, 10]

    // 3. Residual connection + layer norm
    const ffnResidual = matrixAdd(normedAttention, ffnOutput);
    const output = layerNorm(ffnResidual, layerParams.gammaAfterFFN, layerParams.betaAfterFFN);

    return output;
};

const sampleFromDistribution = (probabilities: number[]): number => {
    const random = Math.random();
    let cumulative = 0;
    
    for (let i = 0; i < probabilities.length; i++) {
        cumulative += probabilities[i];
        if (random <= cumulative) {
            return i;
        }
    }
    
    return probabilities.length - 1; // fallback
}

const vectorMatrixMultiply = (vector: number[][], matrix: number[][]): number[][] => {
    const result: number[][] = [];
    for (let i = 0; i < vector.length; i++) {
        result[i] = [];
        for (let j = 0; j < matrix[0].length; j++) {
            let sum = 0;
            for (let k = 0; k < vector[0].length; k++) {
                sum += vector[i][k] * matrix[k][j];
            }
            result[i][j] = sum;
        }
    }
    return result;
}

export const createNewToken = (text: string) => {
    const tokenIndexes = tokenize(text, vocab)
    const embeddingsSequence = createEmbeddingsSequence(tokenIndexes)
    const pozSequence = createPozSequence(embeddingsSequence, positionEmbedding)

    const layer1Output = transformerBlock(pozSequence, {
        Wq: Wq1,
        Wk: Wk1,
        Wv: Wv1,
        Wo: Wo1,
        FFN_W1: FFN_W11,
        FFN_W2: FFN_W21,
        gammaAfterAttention: gammaAfterAttention_1,
        betaAfterAttention: betaAfterAttention_1,
        gammaAfterFFN: gammaAfterFFN_1,
        betaAfterFFN: betaAfterFFN_1
    });

    const layer2Output = transformerBlock(layer1Output, {
        Wq: Wq2,
        Wk: Wk2,
        Wv: Wv2,
        Wo: Wo2,
        FFN_W1: FFN_W12,
        FFN_W2: FFN_W22,
        gammaAfterAttention: gammaAfterAttention_2,
        betaAfterAttention: betaAfterAttention_2,
        gammaAfterFFN: gammaAfterFFN_2,
        betaAfterFFN: betaAfterFFN_2
    });

    const lastTokenEmbedding = layer2Output[tokenIndexes.length - 1];
    const logits = vectorMatrixMultiply([lastTokenEmbedding], linearLayer)[0];
    const temperedLogits = logits.map(logit => logit / TEMPERATURE);
    const probabilities = softmax([temperedLogits])[0];
    const nextTokenIndex = sampleFromDistribution(probabilities);

    return detokenize([nextTokenIndex], vocab);
}