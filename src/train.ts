import { tokenEmbedding } from "./matrix/tokenEmbedding"; //number[][]
import { positionEmbedding } from "./matrix/positionEmbedding"; //number[][]

import { Wk1 } from "./matrix/layer1/Wk1"; //number[][]
import { Wo1 } from "./matrix/layer1/Wo1"; //number[][]
import { Wq1 } from "./matrix/layer1/Wq1"; //number[][]
import { Wv1 } from "./matrix/layer1/Wv1"; //number[][]
import { FFN_W11 } from "./matrix/layer1/FFN_W11"; //number[][]
import { FFN_W21 } from "./matrix/layer1/FFN_W21"; //number[][]
import { betaAfterAttention_1 } from "./matrix/layer1/betaAfterAttention_1"; //number[]
import { gammaAfterAttention_1 } from "./matrix/layer1/gammaAfterAttention_1"; //number[]
import { betaAfterFFN_1 } from "./matrix/layer1/betaAfterFFN_1"; //number[]
import { gammaAfterFFN_1 } from "./matrix/layer1/gammaAfterFFN_1"; //number[]

import { Wk2 } from "./matrix/layer2/Wk2"; //number[][]
import { Wo2 } from "./matrix/layer2/Wo2"; //number[][]
import { Wq2 } from "./matrix/layer2/Wq2"; //number[][]
import { Wv2 } from "./matrix/layer2/Wv2"; //number[][]
import { FFN_W12 } from "./matrix/layer2/FFN_W12"; //number[][]
import { FFN_W22 } from "./matrix/layer2/FFN_W22"; //number[][]
import { betaAfterAttention_2 } from "./matrix/layer2/betaAfterAttention_2"; //number[]
import { gammaAfterAttention_2 } from "./matrix/layer2/gammaAfterAttention_2"; //number[]
import { betaAfterFFN_2 } from "./matrix/layer2/betaAfterFFN_2"; //number[]
import { gammaAfterFFN_2 } from "./matrix/layer2/gammaAfterFFN_2"; //number[]

import { Wk3 } from "./matrix/layer3/Wk3";
import { Wo3 } from "./matrix/layer3/Wo3";
import { Wq3 } from "./matrix/layer3/Wq3";
import { Wv3 } from "./matrix/layer3/Wv3";
import { FFN_W13 } from "./matrix/layer3/FFN_W13";
import { FFN_W23 } from "./matrix/layer3/FFN_W23";
import { betaAfterAttention_3 } from "./matrix/layer3/betaAfterAttention_3";
import { gammaAfterAttention_3 } from "./matrix/layer3/gammaAfterAttention_3";
import { betaAfterFFN_3 } from "./matrix/layer3/betaAfterFFN_3";
import { gammaAfterFFN_3 } from "./matrix/layer3/gammaAfterFFN_3";

import { Wk4 } from "./matrix/layer4/Wk4";
import { Wo4 } from "./matrix/layer4/Wo4";
import { Wq4 } from "./matrix/layer4/Wq4";
import { Wv4 } from "./matrix/layer4/Wv4";
import { FFN_W14 } from "./matrix/layer4/FFN_W14";
import { FFN_W24 } from "./matrix/layer4/FFN_W24";
import { betaAfterAttention_4 } from "./matrix/layer4/betaAfterAttention_4";
import { gammaAfterAttention_4 } from "./matrix/layer4/gammaAfterAttention_4";
import { betaAfterFFN_4 } from "./matrix/layer4/betaAfterFFN_4";
import { gammaAfterFFN_4 } from "./matrix/layer4/gammaAfterFFN_4";

import { linearLayer } from "./matrix/linearLayer"; //number[][]

// ==================== MATRIX OPERATIONS ====================
const matrixMultiply = (a: number[][], b: number[][]): number[][] => {
  const rowsA = a.length;
  const colsA = a[0].length;
  const colsB = b[0].length;

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

const matrixAdd = (a: number[][], b: number[][]): number[][] => {
  return a.map((row, i) => row.map((val, j) => val + b[i][j]));
};

const matrixScalarMultiply = (matrix: number[][], scalar: number): number[][] => {
  return matrix.map(row => row.map(val => val * scalar));
};

// ==================== ACTIVATION FUNCTIONS ====================
const gelu = (x: number): number => {
  return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))));
};

const geluDerivative = (x: number): number => {
  const sqrt2OverPi = Math.sqrt(2 / Math.PI);
  const x3 = 0.044715 * Math.pow(x, 3);
  const tanhArg = sqrt2OverPi * (x + x3);
  const tanhVal = Math.tanh(tanhArg);

  return 0.5 * (1 + tanhVal) + 0.5 * x * (1 - Math.pow(tanhVal, 2)) * sqrt2OverPi * (1 + 3 * 0.044715 * Math.pow(x, 2));
};

// ==================== NEURAL NETWORK OPERATIONS ====================
const softmax = (matrix: number[][]): number[][] => {
  return matrix.map(row => {
    const maxVal = Math.max(...row);
    const expRow = row.map(val => Math.exp(val - maxVal));
    const sumExp = expRow.reduce((sum, val) => sum + val, 0);
    return expRow.map(val => val / sumExp);
  });
};

const layerNorm = (matrix: number[][], gamma: number[], beta: number[]): number[][] => {
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

// ==================== LOSS FUNCTIONS ====================
const computeCrossEntropyLoss = (logits: number[][], targets: number[]): number => {
  let loss = 0;
  for (let i = 0; i < targets.length; i++) {
    const probs = softmax([logits[i]])[0];
    loss += -Math.log(probs[targets[i]]);
  }
  return loss / targets.length;
};

const computeGradientLogits = (logits: number[][], targets: number[]): number[][] => {
  const batchSize = logits.length;
  const vocabSize = logits[0].length;
  const grad = Array(batchSize).fill(0).map(() => Array(vocabSize).fill(0));

  for (let i = 0; i < batchSize; i++) {
    const probs = softmax([logits[i]])[0];
    for (let j = 0; j < vocabSize; j++) {
      grad[i][j] = probs[j] - (j === targets[i] ? 1 : 0);
    }
  }

  return matrixScalarMultiply(grad, 1 / batchSize);
};

// ==================== ATTENTION LAYER ====================
const attentionLayer = (
  input: number[][],
  Wq: number[][],
  Wk: number[][],
  Wv: number[][],
  Wo: number[][]
) => {
  const Q = matrixMultiply(input, Wq);
  const K = matrixMultiply(input, Wk);
  const V = matrixMultiply(input, Wv);

  const Kt = matrixTranspose(K);
  const attentionScores = matrixMultiply(Q, Kt);

  const dk = Q[0].length;
  const scaledAttention = matrixScalarMultiply(attentionScores, 1 / Math.sqrt(dk));

  for (let i = 0; i < scaledAttention.length; i++) {
    for (let j = i + 1; j < scaledAttention[i].length; j++) {
      scaledAttention[i][j] = -Infinity;
    }
  }

  const attentionWeights = softmax(scaledAttention);
  const weightedValues = matrixMultiply(attentionWeights, V);
  const output = matrixMultiply(weightedValues, Wo);

  return {
    output,
    cache: { Q, K, V, attentionScores: scaledAttention, attentionWeights, weightedValues }
  };
};

// ==================== BACKWARD PASS OPERATIONS ====================
const backwardSoftmax = (dOutput: number[][], softmaxOutput: number[][]): number[][] => {
  const batchSize = dOutput.length;
  const result: number[][] = Array(batchSize).fill(0).map(() => Array(dOutput[0].length).fill(0));

  for (let i = 0; i < batchSize; i++) {
    for (let j = 0; j < dOutput[i].length; j++) {
      for (let k = 0; k < dOutput[i].length; k++) {
        const kroneckerDelta = j === k ? 1 : 0;
        result[i][j] += dOutput[i][k] * softmaxOutput[i][k] * (kroneckerDelta - softmaxOutput[i][j]);
      }
    }
  }
  return result;
};

const backwardLayerNorm = (
  dOutput: number[][],
  input: number[][],
  gamma: number[]
): { dInput: number[][], dGamma: number[], dBeta: number[] } => {
  const batchSize = input.length;
  const hiddenSize = input[0].length;
  const epsilon = 1e-8;
  
  const dInput: number[][] = Array(batchSize).fill(0).map(() => Array(hiddenSize).fill(0));
  const dGamma: number[] = Array(hiddenSize).fill(0);
  const dBeta: number[] = Array(hiddenSize).fill(0);

  for (let i = 0; i < batchSize; i++) {
    const mean = input[i].reduce((sum, val) => sum + val, 0) / hiddenSize;
    const variance = input[i].reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / hiddenSize;
    const std = Math.sqrt(variance + epsilon);

    const xMinusMean = input[i].map(val => val - mean);
    const normalized = xMinusMean.map(val => val / std);
    
    for (let j = 0; j < hiddenSize; j++) {
      dBeta[j] += dOutput[i][j];
      dGamma[j] += dOutput[i][j] * normalized[j];
      
      const dNormalized = dOutput[i][j] * gamma[j];
      
      const dVariance = dNormalized * xMinusMean[j] * -0.5 * Math.pow(variance + epsilon, -1.5);
      const dMean = dNormalized * (-1 / std) + dVariance * (-2 * xMinusMean[j]) / hiddenSize;
      
      dInput[i][j] = dNormalized / std + dVariance * 2 * xMinusMean[j] / hiddenSize + dMean / hiddenSize;
    }
  }

  return { dInput, dGamma, dBeta };
};

const backwardAttention = (
  dOutput: number[][],
  input: number[][],
  Wq: number[][],
  Wk: number[][],
  Wv: number[][],
  Wo: number[][],
  cache: any
) => {
  const batchSize = dOutput.length;
  const dk = cache.Q[0].length;

  const dWo = matrixMultiply(matrixTranspose(cache.weightedValues), dOutput);
  const dWeightedValues = matrixMultiply(dOutput, matrixTranspose(Wo));
  const dAttentionWeights = matrixMultiply(dWeightedValues, matrixTranspose(cache.V));
  const dV = matrixMultiply(matrixTranspose(cache.attentionWeights), dWeightedValues);
  const dWv = matrixMultiply(matrixTranspose(input), dV);

  const dScaledScores = backwardSoftmax(dAttentionWeights, cache.attentionWeights);

  for (let i = 0; i < dScaledScores.length; i++) {
    for (let j = i + 1; j < dScaledScores[i].length; j++) {
      dScaledScores[i][j] = 0;
    }
  }

  const dAttentionScores = matrixScalarMultiply(dScaledScores, 1 / Math.sqrt(dk));
  const dQ = matrixMultiply(dAttentionScores, cache.K);
  const dK = matrixMultiply(matrixTranspose(dAttentionScores), cache.Q);
  const dWq = matrixMultiply(matrixTranspose(input), dQ);
  const dWk = matrixMultiply(matrixTranspose(input), dK);

  const dInputFromQ = matrixMultiply(dQ, matrixTranspose(Wq));
  const dInputFromK = matrixMultiply(dK, matrixTranspose(Wk));
  const dInputFromV = matrixMultiply(dV, matrixTranspose(Wv));
  const dInput = matrixAdd(matrixAdd(dInputFromQ, dInputFromK), dInputFromV);

  return { dInput, dWq, dWk, dWv, dWo };
};

const backwardFFN = (
  dOut: number[][],
  input: number[][],
  W1: number[][],
  W2: number[][],
  cache: any
) => {
  const batchSize = dOut.length;
  const hiddenSize = W2.length;
  const outputSize = W2[0].length;

  const dW2: number[][] = Array(hiddenSize).fill(0).map(() => Array(outputSize).fill(0));
  const dHidden: number[][] = Array(batchSize).fill(0).map(() => Array(hiddenSize).fill(0));

  for (let i = 0; i < hiddenSize; i++) {
    for (let j = 0; j < outputSize; j++) {
      let sum = 0;
      for (let k = 0; k < batchSize; k++) {
        sum += cache.ffnHidden[k][i] * dOut[k][j];
      }
      dW2[i][j] = sum;
    }
  }

  for (let i = 0; i < batchSize; i++) {
    for (let j = 0; j < hiddenSize; j++) {
      let sum = 0;
      for (let k = 0; k < outputSize; k++) {
        sum += dOut[i][k] * W2[j][k];
      }
      dHidden[i][j] = sum;
    }
  }

  const dPreActivation: number[][] = Array(batchSize).fill(0).map(() => Array(hiddenSize).fill(0));
  for (let i = 0; i < batchSize; i++) {
    for (let j = 0; j < hiddenSize; j++) {
      dPreActivation[i][j] = dHidden[i][j] * geluDerivative(cache.ffnPreActivation[i][j]);
    }
  }

  const dW1: number[][] = Array(input[0].length).fill(0).map(() => Array(hiddenSize).fill(0));
  const dInput: number[][] = Array(batchSize).fill(0).map(() => Array(input[0].length).fill(0));

  for (let i = 0; i < input[0].length; i++) {
    for (let j = 0; j < hiddenSize; j++) {
      let sum = 0;
      for (let k = 0; k < batchSize; k++) {
        sum += input[k][i] * dPreActivation[k][j];
      }
      dW1[i][j] = sum;
    }
  }

  for (let i = 0; i < batchSize; i++) {
    for (let j = 0; j < input[0].length; j++) {
      let sum = 0;
      for (let k = 0; k < hiddenSize; k++) {
        sum += dPreActivation[i][k] * W1[j][k];
      }
      dInput[i][j] = sum;
    }
  }

  return { dInput, dW1, dW2 };
};

// ==================== TRANSFORMER BLOCK ====================
const transformerBlock = (
  input: number[][],
  layerParams: any
) => {
  const attentionResult = attentionLayer(input, layerParams.Wq, layerParams.Wk, layerParams.Wv, layerParams.Wo);
  const attentionOutput = attentionResult.output;

  const attentionResidual = matrixAdd(input, attentionOutput);
  const normedAttention = layerNorm(attentionResidual, layerParams.gammaAfterAttention, layerParams.betaAfterAttention);

  const ffnPreActivation = matrixMultiply(normedAttention, layerParams.FFN_W1);
  const ffnHidden = ffnPreActivation.map(row => row.map(gelu));
  const ffnOutput = matrixMultiply(ffnHidden, layerParams.FFN_W2);

  const ffnResidual = matrixAdd(normedAttention, ffnOutput);
  const output = layerNorm(ffnResidual, layerParams.gammaAfterFFN, layerParams.betaAfterFFN);

  return {
    output,
    cache: {
      input,
      attentionOutput,
      attentionResidual,
      normedAttention,
      ffnOutput,
      ffnResidual,
      attentionCache: attentionResult.cache,
      ffnHidden,
      ffnPreActivation
    }
  };
};

// ==================== GRADIENT CONTROL ====================
const clipGradientsByValue = (gradients: number[][][], min: number, max: number): void => {
  for (const matrix of gradients) {
    for (let i = 0; i < matrix.length; i++) {
      for (let j = 0; j < matrix[i].length; j++) {
        if (matrix[i][j] < min) matrix[i][j] = min;
        if (matrix[i][j] > max) matrix[i][j] = max;
      }
    }
  }
};

const clipGradientsByNorm = (gradients: number[][][], maxNorm: number): void => {
  let totalNorm = 0;

  for (const matrix of gradients) {
    for (let i = 0; i < matrix.length; i++) {
      for (let j = 0; j < matrix[i].length; j++) {
        totalNorm += matrix[i][j] ** 2;
      }
    }
  }

  totalNorm = Math.sqrt(totalNorm);

  if (totalNorm > maxNorm) {
    const scale = maxNorm / totalNorm;
    for (const matrix of gradients) {
      for (let i = 0; i < matrix.length; i++) {
        for (let j = 0; j < matrix[i].length; j++) {
          matrix[i][j] *= scale;
        }
      }
    }
  }
};

const monitorGradients = (gradients: number[][][], step: number): void => {
  let maxGrad = -Infinity;
  let minGrad = Infinity;
  let avgGrad = 0;
  let count = 0;

  for (const matrix of gradients) {
    for (let i = 0; i < matrix.length; i++) {
      for (let j = 0; j < matrix[i].length; j++) {
        const grad = matrix[i][j];
        maxGrad = Math.max(maxGrad, grad);
        minGrad = Math.min(minGrad, grad);
        avgGrad += Math.abs(grad);
        count++;
      }
    }
  }

  avgGrad /= count;

  console.log(`Step ${step}: Gradients - Max: ${maxGrad.toExponential(2)}, Min: ${minGrad.toExponential(2)}, Avg: ${avgGrad.toExponential(2)}`);

  if (maxGrad > 1000 || minGrad < -1000) {
    console.warn('⚠️  Возможен взрыв градиентов!');
  }
  if (avgGrad < 1e-8) {
    console.warn('⚠️  Возможно затухание градиентов!');
  }
};

// ==================== WEIGHT UPDATES ====================
const updateWeights = (weights: number[][], gradients: number[][], learningRate: number): void => {
  for (let i = 0; i < weights.length; i++) {
    for (let j = 0; j < weights[i].length; j++) {
      weights[i][j] -= learningRate * gradients[i][j];
    }
  }
};

const updateLayerNormParams = (gamma: number[], dGamma: number[], beta: number[], dBeta: number[], learningRate: number): void => {
  for (let i = 0; i < gamma.length; i++) {
    gamma[i] -= learningRate * dGamma[i];
    beta[i] -= learningRate * dBeta[i];
  }
};

// ==================== MAIN TRAINING FUNCTION ====================
let trainingStep = 0;

export const train = (tokensExample: number[], learningRate: number, gradientConfig: { clipValue: { min: number; max: number }; clipNorm: number; monitorFrequency: number; }) => {
  trainingStep++;

  const startIdx = Math.floor(Math.random() * (tokensExample.length - 64));
  const inputTokens = tokensExample.slice(startIdx, startIdx + 64);
  const targetTokens = tokensExample.slice(startIdx + 1, startIdx + 65);

  // Forward pass
  const tokenEmbeddings = inputTokens.map(token => tokenEmbedding[token]);
  const paddingLength = 64 - tokenEmbeddings.length;
  const padding = Array(paddingLength).fill(0).map(() => Array(10).fill(0));

  const paddedTokenEmbeddings = tokenEmbeddings.concat(padding);
  const usedPositionEmbeddings = positionEmbedding.slice(0, 64);
  const combinedEmbeddings = paddedTokenEmbeddings.map((tokenVec, i) =>
    tokenVec.map((val, j) => val + usedPositionEmbeddings[i][j])
  );

  const layer1Result = transformerBlock(combinedEmbeddings, {
    Wq: Wq1, Wk: Wk1, Wv: Wv1, Wo: Wo1,
    FFN_W1: FFN_W11, FFN_W2: FFN_W21,
    gammaAfterAttention: gammaAfterAttention_1, betaAfterAttention: betaAfterAttention_1,
    gammaAfterFFN: gammaAfterFFN_1, betaAfterFFN: betaAfterFFN_1
  });

  const layer2Result = transformerBlock(layer1Result.output, {
    Wq: Wq2, Wk: Wk2, Wv: Wv2, Wo: Wo2,
    FFN_W1: FFN_W12, FFN_W2: FFN_W22,
    gammaAfterAttention: gammaAfterAttention_2, betaAfterAttention: betaAfterAttention_2,
    gammaAfterFFN: gammaAfterFFN_2, betaAfterFFN: betaAfterFFN_2
  });

  const layer3Result = transformerBlock(layer2Result.output, {
    Wq: Wq3, Wk: Wk3, Wv: Wv3, Wo: Wo3,
    FFN_W1: FFN_W13, FFN_W2: FFN_W23,
    gammaAfterAttention: gammaAfterAttention_3, betaAfterAttention: betaAfterAttention_3,
    gammaAfterFFN: gammaAfterFFN_3, betaAfterFFN: betaAfterFFN_3
  });

  const layer4Result = transformerBlock(layer3Result.output, {
    Wq: Wq4, Wk: Wk4, Wv: Wv4, Wo: Wo4,
    FFN_W1: FFN_W14, FFN_W2: FFN_W24,
    gammaAfterAttention: gammaAfterAttention_4, betaAfterAttention: betaAfterAttention_4,
    gammaAfterFFN: gammaAfterFFN_4, betaAfterFFN: betaAfterFFN_4
  });

  const logits = matrixMultiply(layer4Result.output, linearLayer);
  const loss = computeCrossEntropyLoss(logits, targetTokens);

  // Backward pass
  const dLogits = computeGradientLogits(logits, targetTokens);
  const dWLinear = matrixMultiply(matrixTranspose(layer4Result.output), dLogits);
  let dLayer4Output = matrixMultiply(dLogits, matrixTranspose(linearLayer));

  // Layer 4 backward
  const { dInput: dFfnResidual4, dGamma: dGammaAfterFFN4, dBeta: dBetaAfterFFN4 } = backwardLayerNorm(
    dLayer4Output,
    layer4Result.cache.ffnResidual,
    gammaAfterFFN_4
  );

  const dNormedAttention4 = dFfnResidual4;
  const dFfnOutput4 = dFfnResidual4;

  const ffn4Backward = backwardFFN(
    dFfnOutput4,
    layer4Result.cache.normedAttention,
    FFN_W14,
    FFN_W24,
    {
      ffnHidden: layer4Result.cache.ffnHidden,
      ffnPreActivation: layer4Result.cache.ffnPreActivation
    }
  );

  const dAttentionResidual4 = matrixAdd(dNormedAttention4, ffn4Backward.dInput);
  
  const { dInput: dAttentionOutput4, dGamma: dGammaAfterAttention4, dBeta: dBetaAfterAttention4 } = backwardLayerNorm(
    dAttentionResidual4,
    layer4Result.cache.attentionResidual,
    gammaAfterAttention_4
  );

  const dLayer3OutputAfterAttention4 = dAttentionOutput4;
  const dAttentionOutput4FromResidual = dAttentionOutput4;

  const attention4Backward = backwardAttention(
    dAttentionOutput4FromResidual,
    layer4Result.cache.input,
    Wq4, Wk4, Wv4, Wo4,
    layer4Result.cache.attentionCache
  );

  const dLayer3Output = matrixAdd(dLayer3OutputAfterAttention4, attention4Backward.dInput);

  // Layer 3 backward
  const { dInput: dFfnResidual3, dGamma: dGammaAfterFFN3, dBeta: dBetaAfterFFN3 } = backwardLayerNorm(
    dLayer3Output,
    layer3Result.cache.ffnResidual,
    gammaAfterFFN_3
  );

  const dNormedAttention3 = dFfnResidual3;
  const dFfnOutput3 = dFfnResidual3;

  const ffn3Backward = backwardFFN(
    dFfnOutput3,
    layer3Result.cache.normedAttention,
    FFN_W13,
    FFN_W23,
    {
      ffnHidden: layer3Result.cache.ffnHidden,
      ffnPreActivation: layer3Result.cache.ffnPreActivation
    }
  );

  const dAttentionResidual3 = matrixAdd(dNormedAttention3, ffn3Backward.dInput);
  
  const { dInput: dAttentionOutput3, dGamma: dGammaAfterAttention3, dBeta: dBetaAfterAttention3 } = backwardLayerNorm(
    dAttentionResidual3,
    layer3Result.cache.attentionResidual,
    gammaAfterAttention_3
  );

  const dLayer2OutputAfterAttention3 = dAttentionOutput3;
  const dAttentionOutput3FromResidual = dAttentionOutput3;

  const attention3Backward = backwardAttention(
    dAttentionOutput3FromResidual,
    layer3Result.cache.input,
    Wq3, Wk3, Wv3, Wo3,
    layer3Result.cache.attentionCache
  );

  const dLayer2Output = matrixAdd(dLayer2OutputAfterAttention3, attention3Backward.dInput);

  // Layer 2 backward
  const { dInput: dFfnResidual2, dGamma: dGammaAfterFFN2, dBeta: dBetaAfterFFN2 } = backwardLayerNorm(
    dLayer2Output,
    layer2Result.cache.ffnResidual,
    gammaAfterFFN_2
  );

  const dNormedAttention2 = dFfnResidual2;
  const dFfnOutput2 = dFfnResidual2;

  const ffn2Backward = backwardFFN(
    dFfnOutput2,
    layer2Result.cache.normedAttention,
    FFN_W12,
    FFN_W22,
    {
      ffnHidden: layer2Result.cache.ffnHidden,
      ffnPreActivation: layer2Result.cache.ffnPreActivation
    }
  );

  const dAttentionResidual2 = matrixAdd(dNormedAttention2, ffn2Backward.dInput);
  
  const { dInput: dAttentionOutput2, dGamma: dGammaAfterAttention2, dBeta: dBetaAfterAttention2 } = backwardLayerNorm(
    dAttentionResidual2,
    layer2Result.cache.attentionResidual,
    gammaAfterAttention_2
  );

  const dLayer1OutputAfterAttention2 = dAttentionOutput2;
  const dAttentionOutput2FromResidual = dAttentionOutput2;

  const attention2Backward = backwardAttention(
    dAttentionOutput2FromResidual,
    layer2Result.cache.input,
    Wq2, Wk2, Wv2, Wo2,
    layer2Result.cache.attentionCache
  );

  const dLayer1Output = matrixAdd(dLayer1OutputAfterAttention2, attention2Backward.dInput);

  // Layer 1 backward
  const { dInput: dFfnResidual1, dGamma: dGammaAfterFFN1, dBeta: dBetaAfterFFN1 } = backwardLayerNorm(
    dLayer1Output,
    layer1Result.cache.ffnResidual,
    gammaAfterFFN_1
  );

  const dNormedAttention1 = dFfnResidual1;
  const dFfnOutput1 = dFfnResidual1;

  const ffn1Backward = backwardFFN(
    dFfnOutput1,
    layer1Result.cache.normedAttention,
    FFN_W11,
    FFN_W21,
    {
      ffnHidden: layer1Result.cache.ffnHidden,
      ffnPreActivation: layer1Result.cache.ffnPreActivation
    }
  );

  const dAttentionResidual1 = matrixAdd(dNormedAttention1, ffn1Backward.dInput);
  
  const { dInput: dAttentionOutput1, dGamma: dGammaAfterAttention1, dBeta: dBetaAfterAttention1 } = backwardLayerNorm(
    dAttentionResidual1,
    layer1Result.cache.attentionResidual,
    gammaAfterAttention_1
  );

  const dEmbeddingsAfterAttention1 = dAttentionOutput1;
  const dAttentionOutput1FromResidual = dAttentionOutput1;

  const attention1Backward = backwardAttention(
    dAttentionOutput1FromResidual,
    layer1Result.cache.input,
    Wq1, Wk1, Wv1, Wo1,
    layer1Result.cache.attentionCache
  );

  const dEmbeddingsCombined = matrixAdd(dEmbeddingsAfterAttention1, attention1Backward.dInput);

  // Collect all gradients for control
  const allGradients = [
    dWLinear,
    ffn4Backward.dW1, ffn4Backward.dW2,
    attention4Backward.dWq, attention4Backward.dWk, attention4Backward.dWv, attention4Backward.dWo,
    ffn3Backward.dW1, ffn3Backward.dW2,
    attention3Backward.dWq, attention3Backward.dWk, attention3Backward.dWv, attention3Backward.dWo,
    ffn2Backward.dW1, ffn2Backward.dW2,
    attention2Backward.dWq, attention2Backward.dWk, attention2Backward.dWv, attention2Backward.dWo,
    ffn1Backward.dW1, ffn1Backward.dW2,
    attention1Backward.dWq, attention1Backward.dWk, attention1Backward.dWv, attention1Backward.dWo,
    dEmbeddingsCombined
  ];

  // Apply gradient control
  if (gradientConfig) {
    if (gradientConfig.clipValue) {
      clipGradientsByValue(allGradients, gradientConfig.clipValue.min, gradientConfig.clipValue.max);
    }

    if (gradientConfig.clipNorm) {
      clipGradientsByNorm(allGradients, gradientConfig.clipNorm);
    }

    if (gradientConfig.monitorFrequency && trainingStep % gradientConfig.monitorFrequency === 0) {
      monitorGradients(allGradients, trainingStep);
    }
  }

  // Update weights
  updateWeights(linearLayer, dWLinear, learningRate);

  // Layer 4 updates
  updateWeights(FFN_W14, ffn4Backward.dW1, learningRate);
  updateWeights(FFN_W24, ffn4Backward.dW2, learningRate);
  updateWeights(Wq4, attention4Backward.dWq, learningRate);
  updateWeights(Wk4, attention4Backward.dWk, learningRate);
  updateWeights(Wv4, attention4Backward.dWv, learningRate);
  updateWeights(Wo4, attention4Backward.dWo, learningRate);
  updateLayerNormParams(gammaAfterFFN_4, dGammaAfterFFN4, betaAfterFFN_4, dBetaAfterFFN4, learningRate);
  updateLayerNormParams(gammaAfterAttention_4, dGammaAfterAttention4, betaAfterAttention_4, dBetaAfterAttention4, learningRate);

  // Layer 3 updates
  updateWeights(FFN_W13, ffn3Backward.dW1, learningRate);
  updateWeights(FFN_W23, ffn3Backward.dW2, learningRate);
  updateWeights(Wq3, attention3Backward.dWq, learningRate);
  updateWeights(Wk3, attention3Backward.dWk, learningRate);
  updateWeights(Wv3, attention3Backward.dWv, learningRate);
  updateWeights(Wo3, attention3Backward.dWo, learningRate);
  updateLayerNormParams(gammaAfterFFN_3, dGammaAfterFFN3, betaAfterFFN_3, dBetaAfterFFN3, learningRate);
  updateLayerNormParams(gammaAfterAttention_3, dGammaAfterAttention3, betaAfterAttention_3, dBetaAfterAttention3, learningRate);

  // Layer 2 updates
  updateWeights(FFN_W12, ffn2Backward.dW1, learningRate);
  updateWeights(FFN_W22, ffn2Backward.dW2, learningRate);
  updateWeights(Wq2, attention2Backward.dWq, learningRate);
  updateWeights(Wk2, attention2Backward.dWk, learningRate);
  updateWeights(Wv2, attention2Backward.dWv, learningRate);
  updateWeights(Wo2, attention2Backward.dWo, learningRate);
  updateLayerNormParams(gammaAfterFFN_2, dGammaAfterFFN2, betaAfterFFN_2, dBetaAfterFFN2, learningRate);
  updateLayerNormParams(gammaAfterAttention_2, dGammaAfterAttention2, betaAfterAttention_2, dBetaAfterAttention2, learningRate);

  // Layer 1 updates
  updateWeights(FFN_W11, ffn1Backward.dW1, learningRate);
  updateWeights(FFN_W21, ffn1Backward.dW2, learningRate);
  updateWeights(Wq1, attention1Backward.dWq, learningRate);
  updateWeights(Wk1, attention1Backward.dWk, learningRate);
  updateWeights(Wv1, attention1Backward.dWv, learningRate);
  updateWeights(Wo1, attention1Backward.dWo, learningRate);
  updateLayerNormParams(gammaAfterFFN_1, dGammaAfterFFN1, betaAfterFFN_1, dBetaAfterFFN1, learningRate);
  updateLayerNormParams(gammaAfterAttention_1, dGammaAfterAttention1, betaAfterAttention_1, dBetaAfterAttention1, learningRate);

  // Update embeddings
  for (let i = 0; i < inputTokens.length; i++) {
    for (let j = 0; j < 10; j++) {
      tokenEmbedding[inputTokens[i]][j] -= learningRate * dEmbeddingsCombined[i][j];
    }
  }

  for (let i = 0; i < 64; i++) {
    for (let j = 0; j < 10; j++) {
      positionEmbedding[i][j] -= learningRate * dEmbeddingsCombined[i][j];
    }
  }

  return loss;
};