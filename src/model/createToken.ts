import { config } from "../utils/config/config";
import { embeddingMatrix } from "../data/matrix/embeddingMatrix";
import { vocab } from "../utils/config/vocab";
import { positionMatrix } from "../data/matrix/positionMatrix";

// Layer 1 imports
import { gamma1layerFirst } from "../data/matrix/layer_1/gamma1layerFirst";
import { beta1layerFirst } from "../data/matrix/layer_1/beta1layerFirst";
import { gamma1layerSecond } from "../data/matrix/layer_1/gamma1layerSecond";
import { beta1layerSecond } from "../data/matrix/layer_1/beta1layerSecond";
import { W_Q_1layer } from "../data/matrix/layer_1/W_Q_1layer";
import { W_K_1layer } from "../data/matrix/layer_1/W_K_1layer";
import { W_V_1layer } from "../data/matrix/layer_1/W_V_1layer";
import { W_o_1layer } from "../data/matrix/layer_1/W_o_1layer";
import { Weights1_1layer } from "../data/matrix/layer_1/Weights1_1layer";
import { Weights2_1layer } from "../data/matrix/layer_1/Weights2_1layer";
import { biasHidden1layer } from "../data/matrix/layer_1/biasHidden1layer";
import { biasOutput1layer } from "../data/matrix/layer_1/biasOutput1layer";

// Layer 2 imports
import { gamma2layerFirst } from "../data/matrix/layer_2/gamma2layerFirst";
import { beta2layerFirst } from "../data/matrix/layer_2/beta2layerFirst";
import { gamma2layerSecond } from "../data/matrix/layer_2/gamma2layerSecond";
import { beta2layerSecond } from "../data/matrix/layer_2/beta2layerSecond";
import { W_Q_2layer } from "../data/matrix/layer_2/W_Q_2layer";
import { W_K_2layer } from "../data/matrix/layer_2/W_K_2layer";
import { W_V_2layer } from "../data/matrix/layer_2/W_V_2layer";
import { W_o_2layer } from "../data/matrix/layer_2/W_o_2layer";
import { Weights1_2layer } from "../data/matrix/layer_2/Weights1_2layer";
import { Weights2_2layer } from "../data/matrix/layer_2/Weights2_2layer";
import { biasHidden2layer } from "../data/matrix/layer_2/biasHidden2layer";
import { biasOutput2layer } from "../data/matrix/layer_2/biasOutput2layer";

import { linearLayer } from "../data/matrix/linearLayer";

// Constants
const TEMPERATURE = 0.2;
const EPSILON = 1e-8;

// Types
interface TextPreparationResult {
    embeddings: number[][];
    tokenCount: number;
}

// Text Processing
export const normalizeText = (text: string): string => {
    return text.replace(/\s/g, '_').toLowerCase();
};

export const tokenizeText = (text: string): number[] => {
    const normalized = normalizeText(text);
    const tokens: number[] = [];

    for (const char of normalized) {
        const tokenId = vocab.indexOf(char);
        if (tokenId !== -1) {
            tokens.push(tokenId);
        }
    }

    return tokens;
};

const calculatePositionalEmbeddings = (embeddings: number[][]): number[][] => {
    const positionalEmbeddings: number[][] = [];

    for (let i = 0; i < config.contextLength; i++) {
        const row: number[] = [];
        for (let j = 0; j < config.embeddingSize; j++) {
            row.push(embeddings[i][j] + positionMatrix[i][j]);
        }
        positionalEmbeddings.push(row);
    }

    return positionalEmbeddings;
};

export const prepareText = (text: string): TextPreparationResult => {
    const tokens = tokenizeText(text);
    const embeddings: number[][] = [];

    // Create token embeddings
    for (const tokenId of tokens) {
        embeddings.push([...embeddingMatrix[tokenId]]);
    }

    // Add padding
    const paddingCount = config.contextLength - tokens.length;
    for (let i = 0; i < paddingCount; i++) {
        embeddings.push(new Array(config.embeddingSize).fill(0));
    }

    // Add positional embeddings
    const positionalEmbeddings = calculatePositionalEmbeddings(embeddings);

    return {
        embeddings: positionalEmbeddings,
        tokenCount: tokens.length
    };
};

// Matrix Operations
export const matrixMultiply = (A: number[][], B: number[][]): number[][] => {
    const rowsA = A.length;
    const colsA = A[0].length;
    const colsB = B[0].length;

    if (colsA !== B.length) {
        throw new Error("Incompatible matrix dimensions for multiplication");
    }

    const result: number[][] = Array(rowsA);

    for (let i = 0; i < rowsA; i++) {
        result[i] = Array(colsB).fill(0);
        for (let j = 0; j < colsB; j++) {
            for (let k = 0; k < colsA; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return result;
};

export const matrixTranspose = (matrix: number[][]): number[][] => {
    const rows = matrix.length;
    const cols = matrix[0].length;
    const result: number[][] = Array(cols);

    for (let j = 0; j < cols; j++) {
        result[j] = Array(rows);
        for (let i = 0; i < rows; i++) {
            result[j][i] = matrix[i][j];
        }
    }

    return result;
};

export const matrixAdd = (A: number[][], B: number[][]): number[][] => {
    return A.map((row, i) => row.map((val, j) => val + B[i][j]));
};

export const addBias = (matrix: number[][], bias: number[]): number[][] => {
    return matrix.map(row => row.map((val, j) => val + bias[j]));
};

// Neural Network Operations
export const layerNorm = (
    input: number[][],
    gamma: number[],
    beta: number[]
): number[][] => {
    return input.map(row => {
        const mean = row.reduce((sum, val) => sum + val, 0) / row.length;
        const variance = row.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / row.length;
        const stdDev = Math.sqrt(variance + EPSILON);

        return row.map((val, j) => {
            const normalized = (val - mean) / stdDev;
            return normalized * gamma[j] + beta[j];
        });
    });
};

export const softmax = (matrix: number[][]): number[][] => {
    return matrix.map(row => {
        const maxVal = Math.max(...row);
        const expRow = row.map(val => Math.exp(val - maxVal));
        const sum = expRow.reduce((a, b) => a + b, 0);
        return expRow.map(expVal => expVal / sum);
    });
};

export const gelu = (x: number): number => {
    return 0.5 * x * (1.79788456 * (x + 0.044715 * Math.pow(x, 3)));
};

export const applyActivation = (
    matrix: number[][],
    activationFn: (x: number) => number
): number[][] => {
    return matrix.map(row => row.map(activationFn));
};

// Attention Mechanism
const createCausalMask = (sequenceLength: number): number[][] => {
    const mask: number[][] = Array(sequenceLength);

    for (let i = 0; i < sequenceLength; i++) {
        mask[i] = Array(sequenceLength);
        for (let j = 0; j < sequenceLength; j++) {
            mask[i][j] = j > i ? -1e9 : 0;
        }
    }

    return mask;
};

export const attention = (
    embeddings: number[][],
    gamma1: number[],
    beta1: number[],
    gamma2: number[],
    beta2: number[],
    W_Q: number[][],
    W_K: number[][],
    W_V: number[][],
    W_O: number[][]
): number[][] => {
    // Layer normalization
    const normalized = layerNorm(embeddings, gamma1, beta1);

    // Query, Key, Value projections
    const Q = matrixMultiply(normalized, W_Q);
    const K = matrixMultiply(normalized, W_K);
    const V = matrixMultiply(normalized, W_V);

    // Attention scores
    const K_T = matrixTranspose(K);
    let scores = matrixMultiply(Q, K_T);

    // Scale scores
    const d_k = Q[0].length;
    const scaleFactor = Math.sqrt(d_k);
    scores = scores.map(row => row.map(val => val / scaleFactor));

    // Apply causal mask
    const mask = createCausalMask(scores.length);
    scores = matrixAdd(scores, mask);

    // Softmax and weighted sum
    const attentionWeights = softmax(scores);
    const attentionOutput = matrixMultiply(attentionWeights, V);

    // Output projection and residual connection
    const projectedOutput = matrixMultiply(attentionOutput, W_O);
    const residualOutput = matrixAdd(embeddings, projectedOutput);

    // Final layer normalization
    return layerNorm(residualOutput, gamma2, beta2);
};

// Feed-Forward Network
export const feedForwardNetwork = (
    input: number[][],
    weights1: number[][],
    weights2: number[][],
    biasHidden: number[],
    biasOutput: number[]
): number[][] => {
    const hidden = matrixMultiply(input, weights1);
    const hiddenWithBias = addBias(hidden, biasHidden);
    const activated = applyActivation(hiddenWithBias, gelu);
    const output = matrixMultiply(activated, weights2);

    return addBias(output, biasOutput);
};

// Prediction Utilities
export const calculateLogits = (input: number[], weights: number[][]): number[] => {
    const numTokens = weights[0].length;
    const logits: number[] = new Array(numTokens).fill(0);

    for (let tokenIdx = 0; tokenIdx < numTokens; tokenIdx++) {
        for (let embedIdx = 0; embedIdx < input.length; embedIdx++) {
            logits[tokenIdx] += weights[embedIdx][tokenIdx] * input[embedIdx];
        }
    }

    return logits;
};

export const softmaxWithTemperature = (
    logits: number[],
    temperature: number = TEMPERATURE
): number[] => {
    const scaled = logits.map(x => x / temperature);
    const maxLogit = Math.max(...scaled);
    const exps = scaled.map(x => Math.exp(x - maxLogit));
    const sum = exps.reduce((a, b) => a + b, 0);

    return exps.map(exp => exp / sum);
};

export const sampleFromDistribution = (probabilities: number[]): string => {
    const random = Math.random();
    let cumulativeProb = 0;

    for (let i = 0; i < probabilities.length; i++) {
        cumulativeProb += probabilities[i];
        if (random <= cumulativeProb) {
            return vocab[i];
        }
    }

    // Fallback to highest probability token
    const maxIndex = probabilities.indexOf(Math.max(...probabilities));
    return vocab[maxIndex];
};

export const predictNextToken = (
    ffnOutput: number[][],
    contextLength: number,
    linearWeights: number[][],
    temperature: number = TEMPERATURE
): string => {
    const lastPositionOutput = ffnOutput[contextLength - 1];
    const logits = calculateLogits(lastPositionOutput, linearWeights);
    const probabilities = softmaxWithTemperature(logits, temperature);

    return sampleFromDistribution(probabilities);
};

// Main Inference Function
export const generateNextToken = (text: string): string => {
    const { embeddings, tokenCount } = prepareText(text);

    // First transformer layer
    const attention1 = attention(
        embeddings,
        gamma1layerFirst,
        beta1layerFirst,
        gamma1layerSecond,
        beta1layerSecond,
        W_Q_1layer,
        W_K_1layer,
        W_V_1layer,
        W_o_1layer
    );

    const ffn1 = feedForwardNetwork(
        attention1,
        Weights1_1layer,
        Weights2_1layer,
        biasHidden1layer,
        biasOutput1layer
    );

    // Second transformer layer
    const attention2 = attention(
        ffn1,
        gamma2layerFirst,
        beta2layerFirst,
        gamma2layerSecond,
        beta2layerSecond,
        W_Q_2layer,
        W_K_2layer,
        W_V_2layer,
        W_o_2layer
    );

    const ffn2 = feedForwardNetwork(
        attention2,
        Weights1_2layer,
        Weights2_2layer,
        biasHidden2layer,
        biasOutput2layer
    );

    // Final prediction
    return predictNextToken(ffn2, tokenCount, linearLayer);
};

// Export constants for external configuration
export { TEMPERATURE, EPSILON };