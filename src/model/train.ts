import { embeddingMatrix } from "../data/matrix/embeddingMatrix"
import { positionMatrix } from "../data/matrix/positionMatrix"

// Layer 1 imports
import { gamma1layerFirst } from "../data/matrix/layer_1/gamma1layerFirst"
import { beta1layerFirst } from "../data/matrix/layer_1/beta1layerFirst"
import { gamma1layerSecond } from "../data/matrix/layer_1/gamma1layerSecond"
import { beta1layerSecond } from "../data/matrix/layer_1/beta1layerSecond"
import { W_Q_1layer } from "../data/matrix/layer_1/W_Q_1layer"
import { W_K_1layer } from "../data/matrix/layer_1/W_K_1layer"
import { W_V_1layer } from "../data/matrix/layer_1/W_V_1layer"
import { W_o_1layer } from "../data/matrix/layer_1/W_o_1layer"
import { Weights1_1layer } from "../data/matrix/layer_1/Weights1_1layer"
import { Weights2_1layer } from "../data/matrix/layer_1/Weights2_1layer"
import { biasHidden1layer } from "../data/matrix/layer_1/biasHidden1layer"
import { biasOutput1layer } from "../data/matrix/layer_1/biasOutput1layer"

// Layer 2 imports
import { gamma2layerFirst } from "../data/matrix/layer_2/gamma2layerFirst"
import { beta2layerFirst } from "../data/matrix/layer_2/beta2layerFirst"
import { gamma2layerSecond } from "../data/matrix/layer_2/gamma2layerSecond"
import { beta2layerSecond } from "../data/matrix/layer_2/beta2layerSecond"
import { W_Q_2layer } from "../data/matrix/layer_2/W_Q_2layer"
import { W_K_2layer } from "../data/matrix/layer_2/W_K_2layer"
import { W_V_2layer } from "../data/matrix/layer_2/W_V_2layer"
import { W_o_2layer } from "../data/matrix/layer_2/W_o_2layer"
import { Weights1_2layer } from "../data/matrix/layer_2/Weights1_2layer"
import { Weights2_2layer } from "../data/matrix/layer_2/Weights2_2layer"
import { biasHidden2layer } from "../data/matrix/layer_2/biasHidden2layer"
import { biasOutput2layer } from "../data/matrix/layer_2/biasOutput2layer"

import { linearLayer } from "../data/matrix/linearLayer"
import { config } from "../utils/config/config"

// Constants
const VOCAB_SIZE = 65;
const CONTEXT_LENGTH = 64;
const EMBEDDING_DIM = 10;
const EPSILON = 1e-8;
const MAX_GRAD_NORM = 2;

// Types
interface LayerNormCache {
    input: number[][];
    gamma: number[];
    beta: number[];
    mean: number[];
    variance: number[];
    stdDev: number[];
}

interface SoftmaxCache {
    input: number[][];
    maxVal: number[];
    expSum: number[];
}

interface AttentionCache {
    norm1: number[][];
    norm1Cache: LayerNormCache;
    Q: number[][];
    K: number[][];
    V: number[][];
    W_Q: number[][];
    W_K: number[][];
    W_V: number[][];
    scoresBeforeScale: number[][];
    scoresAfterScale: number[][];
    mask: number[][];
    scoresAfterMask: number[][];
    weights: number[][];
    softmaxCache: SoftmaxCache;
    attentionOutput: number[][];
    projOutput: number[][];
    W_O: number[][];
    residualInput: number[][];
    norm2: number[][];
    norm2Cache: LayerNormCache;
}

interface FFNCache {
    input: number[][];
    hidden: number[][];
    hiddenWithBias: number[][];
    activated: number[][];
    outputBeforeBias: number[][];
}

// Matrix Operations
const matrixMultiply = (A: number[][], B: number[][]): number[][] => {
    const rowsA = A.length, colsA = A[0].length, colsB = B[0].length;
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
}

const matrixTranspose = (matrix: number[][]): number[][] => {
    const rows = matrix.length, cols = matrix[0].length;
    const result: number[][] = Array(cols);
    
    for (let j = 0; j < cols; j++) {
        result[j] = Array(rows);
        for (let i = 0; i < rows; i++) {
            result[j][i] = matrix[i][j];
        }
    }
    return result;
}

const matrixAdd = (A: number[][], B: number[][]): number[][] => 
    A.map((row, i) => row.map((val, j) => val + B[i][j]));

const addBias = (matrix: number[][], bias: number[]): number[][] =>
    matrix.map(row => row.map((val, j) => val + bias[j]));

// Data Preparation
const getTrainingExample = (tokens: number[]) => {
    const maxStartIndex = tokens.length - CONTEXT_LENGTH;
    const startIndex = Math.floor(Math.random() * (maxStartIndex + 1));
    const exampleArray = tokens.slice(startIndex, startIndex + CONTEXT_LENGTH);
    
    const embeddings: number[][] = exampleArray.map(tokenId => embeddingMatrix[tokenId]);
    
    const embeddingsWithPosition = embeddings.map((embedding, i) => 
        embedding.map((val, j) => val + positionMatrix[i][j])
    );
    
    return { 
        embeddingsVector: embeddingsWithPosition, 
        embeddingsIndex: exampleArray 
    };
}

// Normalization
const layerNormWithCache = (input: number[][], gamma: number[], beta: number[]): { output: number[][], cache: LayerNormCache } => {
    const cache: LayerNormCache = {
        input: input.map(row => [...row]),
        gamma: [...gamma],
        beta: [...beta],
        mean: [],
        variance: [],
        stdDev: []
    };

    const output = input.map((row, i) => {
        const mean = row.reduce((sum, val) => sum + val, 0) / row.length;
        const variance = row.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / row.length;
        const stdDev = Math.sqrt(variance + EPSILON);

        cache.mean[i] = mean;
        cache.variance[i] = variance;
        cache.stdDev[i] = stdDev;

        return row.map((val, j) => {
            const normalized = (val - mean) / stdDev;
            return normalized * gamma[j] + beta[j];
        });
    });

    return { output, cache };
};

// Activation Functions
const gelu = (x: number): number => 
    0.5 * x * (1.79788456 * (x + 0.044715 * Math.pow(x, 3)));

const geluDerivative = (x: number): number => {
    const x3 = x * x * x;
    return 0.89894228 * (2 * x + 0.17886 * x3);
}

// Masking
const createCausalMask = (sequenceLength: number): number[][] => 
    Array.from({ length: sequenceLength }, (_, i) =>
        Array.from({ length: sequenceLength }, (_, j) => 
            j > i ? -1e9 : 0
        )
    );

// Softmax
const softmaxWithCache = (matrix: number[][]): { output: number[][], cache: SoftmaxCache } => {
    const cache: SoftmaxCache = {
        input: matrix.map(row => [...row]),
        maxVal: [],
        expSum: []
    };

    const output = matrix.map((row, i) => {
        const maxVal = Math.max(...row);
        cache.maxVal[i] = maxVal;

        const expRow = row.map(val => Math.exp(val - maxVal));
        const expSum = expRow.reduce((sum, val) => sum + val, 0);
        cache.expSum[i] = expSum;

        return expRow.map(expVal => expVal / expSum);
    });

    return { output, cache };
};

const softmax = (row: number[]): number[] => {
    const max = Math.max(...row);
    const exps = row.map(x => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(x => x / sum);
};

// Attention Mechanism
const attentionForward = (
    embeddings: number[][],
    gamma1: number[],
    beta1: number[],
    gamma2: number[],
    beta2: number[],
    W_Q: number[][],
    W_K: number[][],
    W_V: number[][],
    W_O: number[][]
): { output: number[][], cache: AttentionCache } => {
    const cache: Partial<AttentionCache> = {};

    // Layer normalization 1
    const norm1 = layerNormWithCache(embeddings, gamma1, beta1);
    cache.norm1 = norm1.output;
    cache.norm1Cache = norm1.cache;

    // Query, Key, Value projections
    cache.Q = matrixMultiply(norm1.output, W_Q);
    cache.K = matrixMultiply(norm1.output, W_K);
    cache.V = matrixMultiply(norm1.output, W_V);
    cache.W_Q = W_Q;
    cache.W_K = W_K;
    cache.W_V = W_V;

    // Attention scores
    let scores = matrixMultiply(cache.Q, matrixTranspose(cache.K));
    cache.scoresBeforeScale = scores.map(row => [...row]);

    // Scaling
    const d_k = cache.Q[0].length;
    scores = scores.map(row => row.map(val => val / Math.sqrt(d_k)));
    cache.scoresAfterScale = scores.map(row => [...row]);

    // Causal masking
    const mask = createCausalMask(scores.length);
    cache.mask = mask;
    scores = scores.map((row, i) => row.map((val, j) => val + mask[i][j]));
    cache.scoresAfterMask = scores.map(row => [...row]);

    // Softmax
    const weights = softmaxWithCache(scores);
    cache.weights = weights.output;
    cache.softmaxCache = weights.cache;

    // Attention output and projection
    cache.attentionOutput = matrixMultiply(weights.output, cache.V);
    cache.projOutput = matrixMultiply(cache.attentionOutput, W_O);
    cache.W_O = W_O;

    // Residual connection and layer normalization 2
    cache.residualInput = embeddings;
    const residual = matrixAdd(embeddings, cache.projOutput);
    const norm2 = layerNormWithCache(residual, gamma2, beta2);
    cache.norm2 = norm2.output;
    cache.norm2Cache = norm2.cache;

    return { output: norm2.output, cache: cache as AttentionCache };
};

// Feed Forward Network
const applyActivation = (matrix: number[][], activationFn: (x: number) => number): number[][] =>
    matrix.map(row => row.map(activationFn));

const FFNForward = (
    input: number[][],
    weights1: number[][],
    weights2: number[][],
    biasHidden: number[],
    biasOutput: number[]
): { output: number[][], cache: FFNCache } => {
    const cache: FFNCache = { input } as FFNCache;
    
    cache.hidden = matrixMultiply(input, weights1);
    cache.hiddenWithBias = addBias(cache.hidden, biasHidden);
    cache.activated = applyActivation(cache.hiddenWithBias, gelu);
    cache.outputBeforeBias = matrixMultiply(cache.activated, weights2);
    
    const output = addBias(cache.outputBeforeBias, biasOutput);
    return { output, cache };
};

// Linear Layer
const linearForward = (input: number[][], weights: number[][]): number[][] => {
    const output: number[][] = Array(CONTEXT_LENGTH);
    
    for (let i = 0; i < CONTEXT_LENGTH; i++) {
        output[i] = Array(VOCAB_SIZE).fill(0);
        for (let j = 0; j < VOCAB_SIZE; j++) {
            for (let k = 0; k < EMBEDDING_DIM; k++) {
                output[i][j] += input[i][k] * weights[k][j];
            }
        }
    }
    return output;
};

// Loss Calculation
const calculateLoss = (logits: number[][], targets: number[]): number => {
    let loss = 0;
    for (let i = 0; i < CONTEXT_LENGTH - 1; i++) {
        const probs = softmax(logits[i]);
        loss += -Math.log(probs[targets[i + 1]] + EPSILON);
    }
    return loss / (CONTEXT_LENGTH - 1);
};

// Gradient Clipping
const clipGradientsVector = (grad: number[], maxNorm: number = MAX_GRAD_NORM): number[] => {
    const norm = Math.sqrt(grad.reduce((sum, val) => sum + val * val, 0));
    return norm > maxNorm ? grad.map(val => val * maxNorm / norm) : grad;
};

const clipGradients = (grad: number[][], maxNorm: number = MAX_GRAD_NORM): number[][] => {
    const norm = Math.sqrt(grad.flat().reduce((sum, val) => sum + val * val, 0));
    return norm > maxNorm ? grad.map(row => row.map(val => val * maxNorm / norm)) : grad;
};

// Backward Pass Functions
const softmaxCrossEntropyBackward = (logits: number[][], targets: number[]): number[][] => {
    const gradOutput: number[][] = Array(CONTEXT_LENGTH).fill(0).map(() => Array(VOCAB_SIZE).fill(0));
    
    for (let i = 0; i < CONTEXT_LENGTH - 1; i++) {
        const probs = softmax(logits[i]);
        for (let j = 0; j < VOCAB_SIZE; j++) {
            gradOutput[i][j] = probs[j] - (j === targets[i + 1] ? 1 : 0);
        }
    }
    return gradOutput;
};

const linearBackward = (
    input: number[][],
    gradOutput: number[][],
    weights: number[][]
): { gradWeights: number[][], gradInput: number[][] } => {
    const gradWeights: number[][] = Array(EMBEDDING_DIM).fill(0).map(() => Array(VOCAB_SIZE).fill(0));
    const gradInput: number[][] = Array(CONTEXT_LENGTH).fill(0).map(() => Array(EMBEDDING_DIM).fill(0));

    // Compute weight gradients
    for (let i = 0; i < CONTEXT_LENGTH; i++) {
        for (let j = 0; j < VOCAB_SIZE; j++) {
            for (let k = 0; k < EMBEDDING_DIM; k++) {
                gradWeights[k][j] += input[i][k] * gradOutput[i][j];
            }
        }
    }

    // Compute input gradients
    for (let i = 0; i < CONTEXT_LENGTH; i++) {
        for (let j = 0; j < EMBEDDING_DIM; j++) {
            for (let k = 0; k < VOCAB_SIZE; k++) {
                gradInput[i][j] += weights[j][k] * gradOutput[i][k];
            }
        }
    }

    return { gradWeights, gradInput };
};

const updateWeights = (weights: number[][], gradWeights: number[][], learningRate: number): void => {
    for (let i = 0; i < weights.length; i++) {
        for (let j = 0; j < weights[i].length; j++) {
            weights[i][j] -= learningRate * gradWeights[i][j];
        }
    }
};

// Layer Norm Backward
const layerNormBackward = (dOutput: number[][], cache: LayerNormCache) => {
    const { input, gamma, mean, variance, stdDev } = cache;
    const dGamma = Array(gamma.length).fill(0);
    const dBeta = Array(gamma.length).fill(0);
    const dInput: number[][] = Array(input.length).fill(0).map(() => Array(input[0].length).fill(0));

    for (let i = 0; i < input.length; i++) {
        const xMinusMean = input[i].map(x => x - mean[i]);
        const invStdDev = 1 / stdDev[i];

        // Gradients for gamma and beta
        for (let j = 0; j < input[0].length; j++) {
            dGamma[j] += dOutput[i][j] * xMinusMean[j] * invStdDev;
            dBeta[j] += dOutput[i][j];
        }

        // Gradient for input
        const dxHat = dOutput[i].map((val, j) => val * gamma[j]);
        const dVar = dxHat.reduce((sum, val, j) => 
            sum + val * xMinusMean[j] * -0.5 * Math.pow(variance[i] + EPSILON, -1.5), 0);
        const dMean = dxHat.reduce((sum, val) => sum + val * -invStdDev, 0) + 
                     dVar * -2 * xMinusMean.reduce((sum, val) => sum + val, 0) / input[0].length;

        for (let j = 0; j < input[0].length; j++) {
            dInput[i][j] = dxHat[j] * invStdDev +
                          dVar * 2 * xMinusMean[j] / input[0].length +
                          dMean / input[0].length;
        }
    }

    return { 
        dInput: clipGradients(dInput), 
        dGamma: clipGradientsVector(dGamma), 
        dBeta: clipGradientsVector(dBeta) 
    };
};

// Softmax Backward
const softmaxBackward = (dOutput: number[][], cache: SoftmaxCache): number[][] => {
    const { input, maxVal, expSum } = cache;
    const dInput: number[][] = Array(input.length).fill(0).map(() => Array(input[0].length).fill(0));

    for (let i = 0; i < input.length; i++) {
        const expRow = input[i].map(val => Math.exp(val - maxVal[i]));
        const outputRow = expRow.map(expVal => expVal / expSum[i]);
        
        const sum = dOutput[i].reduce((total, grad, j) => total + grad * outputRow[j], 0);
        
        for (let j = 0; j < input[0].length; j++) {
            dInput[i][j] = outputRow[j] * (dOutput[i][j] - sum);
        }
    }

    return clipGradients(dInput);
};

// Attention Backward
const attentionBackward = (dOutput: number[][], cache: AttentionCache, learningRate: number = 0.01) => {
    // Layer norm 2 backward
    const { dInput: dNorm2, dGamma: dGamma2, dBeta: dBeta2 } = layerNormBackward(dOutput, cache.norm2Cache);

    // Residual connection
    const dProjOutput = dNorm2;

    // Output projection backward
    const dAttentionOutput = matrixMultiply(dProjOutput, matrixTranspose(cache.W_O));
    const dW_O = clipGradients(matrixMultiply(matrixTranspose(cache.attentionOutput), dProjOutput));

    // Attention weights backward
    const dWeights = matrixMultiply(dAttentionOutput, matrixTranspose(cache.V));
    const dV = matrixMultiply(matrixTranspose(cache.weights), dAttentionOutput);

    // Softmax backward
    const dScoresAfterMask = softmaxBackward(dWeights, cache.softmaxCache);

    // Mask and scaling backward
    const dScoresBeforeScale = dScoresAfterMask.map(row => 
        row.map(val => val / Math.sqrt(cache.Q[0].length))
    );

    // Attention scores backward
    const dQ = matrixMultiply(dScoresBeforeScale, cache.K);
    const dK = matrixMultiply(matrixTranspose(dScoresBeforeScale), cache.Q);

    // Query, Key, Value projections backward
    const dNorm1 = matrixAdd(
        matrixMultiply(dQ, matrixTranspose(cache.W_Q)),
        matrixAdd(
            matrixMultiply(dK, matrixTranspose(cache.W_K)),
            matrixMultiply(dV, matrixTranspose(cache.W_V))
        )
    );

    const dW_Q = clipGradients(matrixMultiply(matrixTranspose(cache.norm1), dQ));
    const dW_K = clipGradients(matrixMultiply(matrixTranspose(cache.norm1), dK));
    const dW_V = clipGradients(matrixMultiply(matrixTranspose(cache.norm1), dV));

    // Layer norm 1 backward
    const { dInput: dFirstNorm, dGamma: dGamma1, dBeta: dBeta1 } = layerNormBackward(dNorm1, cache.norm1Cache);

    // Combine gradients
    const dInput = matrixAdd(dFirstNorm, dNorm2);

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

// FFN Backward
const FFNBackward = (
    dOutput: number[][],
    cache: FFNCache,
    weights1: number[][],
    weights2: number[][],
    learningRate: number = 0.01
) => {
    const { input, hidden, hiddenWithBias, activated, outputBeforeBias } = cache;

    // Output bias gradients
    const db2 = dOutput.reduce((sum, row) => 
        row.map((val, j) => (sum[j] || 0) + val), Array(weights2[0].length).fill(0)
    );

    // Output linear transformation gradients
    const dActivated = matrixMultiply(dOutput, matrixTranspose(weights2));
    const dW2 = matrixMultiply(matrixTranspose(activated), dOutput);

    // GELU activation gradients
    const dHiddenWithBias = dActivated.map((row, i) => 
        row.map((val, j) => val * geluDerivative(hiddenWithBias[i][j]))
    );

    // Hidden bias gradients
    const db1 = dHiddenWithBias.reduce((sum, row) => 
        row.map((val, j) => (sum[j] || 0) + val), Array(weights1[0].length).fill(0)
    );

    // Hidden linear transformation gradients
    const dInput = matrixMultiply(dHiddenWithBias, matrixTranspose(weights1));
    const dW1 = matrixMultiply(matrixTranspose(input), dHiddenWithBias);

    return { 
        dInput, 
        dW1: clipGradients(dW1), 
        db1: clipGradientsVector(db1), 
        dW2: clipGradients(dW2), 
        db2: clipGradientsVector(db2) 
    };
};

// Weight Update Functions
const updateFFNWeights = (
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
    updateWeights(weights1, dW1, learningRate);
    updateWeights(weights2, dW2, learningRate);
    
    for (let j = 0; j < biasHidden.length; j++) {
        biasHidden[j] -= learningRate * db1[j];
    }
    for (let j = 0; j < biasOutput.length; j++) {
        biasOutput[j] -= learningRate * db2[j];
    }
};

const updateAttentionWeights = (
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
    updateWeights(W_Q, dW_Q, learningRate);
    updateWeights(W_K, dW_K, learningRate);
    updateWeights(W_V, dW_V, learningRate);
    updateWeights(W_O, dW_O, learningRate);
    
    for (let i = 0; i < gamma1.length; i++) {
        gamma1[i] -= learningRate * dGamma1[i];
        beta1[i] -= learningRate * dBeta1[i];
    }
    for (let i = 0; i < gamma2.length; i++) {
        gamma2[i] -= learningRate * dGamma2[i];
        beta2[i] -= learningRate * dBeta2[i];
    }
};

const updateEmbeddingsAndPositions = (
    gradInput: number[][], 
    tokens: number[], 
    learningRate: number
): void => {
    for (let pos = 0; pos < tokens.length; pos++) {
        const tokenId = tokens[pos];
        for (let dim = 0; dim < config.embeddingSize; dim++) {
            embeddingMatrix[tokenId][dim] -= learningRate * gradInput[pos][dim];
            positionMatrix[pos][dim] -= learningRate * gradInput[pos][dim];
        }
    }
};

// Main Training Function
const updateLinearLayer = (input: number[][], tokens: number[], learningRate: number) => {
    const logits = linearForward(input, linearLayer);
    const loss = calculateLoss(logits, tokens);
    const gradOutput = softmaxCrossEntropyBackward(logits, tokens);
    const { gradWeights, gradInput } = linearBackward(input, gradOutput, linearLayer);
    
    updateWeights(linearLayer, gradWeights, learningRate);
    return { gradInput, loss };
};

export const train = (tokens: number[], learningRate: number = 0.0002): number => {
    // Forward pass
    const { embeddingsVector, embeddingsIndex } = getTrainingExample(tokens);
    const { output: att1, cache: att1Cache } = attentionForward(embeddingsVector, gamma1layerFirst, beta1layerFirst, gamma1layerSecond, beta1layerSecond, W_Q_1layer, W_K_1layer, W_V_1layer, W_o_1layer);
    const { output: FFN1, cache: FFN1Cache } = FFNForward(att1, Weights1_1layer, Weights2_1layer, biasHidden1layer, biasOutput1layer);
    const { output: att2, cache: att2Cache } = attentionForward(FFN1, gamma2layerFirst, beta2layerFirst, gamma2layerSecond, beta2layerSecond, W_Q_2layer, W_K_2layer, W_V_2layer, W_o_2layer);
    const { output: FFN2, cache: FFN2Cache } = FFNForward(att2, Weights1_2layer, Weights2_2layer, biasHidden2layer, biasOutput2layer);

    // Backward pass
    const { gradInput, loss } = updateLinearLayer(FFN2, embeddingsIndex, learningRate);
    const { dInput: dFFN2Input, dW1: dW1_2, db1: db1_2, dW2: dW2_2, db2: db2_2 } = 
    FFNBackward(gradInput, FFN2Cache, Weights1_2layer, Weights2_2layer, learningRate);
    updateFFNWeights(Weights1_2layer, Weights2_2layer, biasHidden2layer, biasOutput2layer, dW1_2, dW2_2, db1_2, db2_2, learningRate);
    const attentionGrads2 = attentionBackward(dFFN2Input, att2Cache, learningRate);
    updateAttentionWeights(W_Q_2layer, W_K_2layer, W_V_2layer, W_o_2layer, gamma2layerFirst, beta2layerFirst, gamma2layerSecond, beta2layerSecond, attentionGrads2.dW_Q, attentionGrads2.dW_K, attentionGrads2.dW_V, attentionGrads2.dW_O, attentionGrads2.dGamma1, attentionGrads2.dBeta1, attentionGrads2.dGamma2, attentionGrads2.dBeta2, learningRate);
    const { dInput: dFFN1Input, dW1: dW1_1, db1: db1_1, dW2: dW2_1, db2: db2_1 } = FFNBackward(attentionGrads2.dInput, FFN1Cache, Weights1_1layer, Weights2_1layer, learningRate);
    updateFFNWeights(Weights1_1layer, Weights2_1layer, biasHidden1layer, biasOutput1layer, dW1_1, dW2_1, db1_1, db2_1, learningRate);
    const attentionGrads1 = attentionBackward(dFFN1Input, att1Cache, learningRate);
    updateAttentionWeights(W_Q_1layer, W_K_1layer, W_V_1layer, W_o_1layer, gamma1layerFirst, beta1layerFirst, gamma1layerSecond, beta1layerSecond, attentionGrads1.dW_Q, attentionGrads1.dW_K, attentionGrads1.dW_V, attentionGrads1.dW_O, attentionGrads1.dGamma1, attentionGrads1.dBeta1, attentionGrads1.dGamma2, attentionGrads1.dBeta2, learningRate);
    updateEmbeddingsAndPositions(attentionGrads1.dInput, embeddingsIndex, learningRate);
    console.log(loss);
    return loss;
};