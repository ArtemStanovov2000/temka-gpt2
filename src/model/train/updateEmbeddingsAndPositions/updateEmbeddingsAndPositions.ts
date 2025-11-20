import { config } from "../../../utils/config/config";

export const updateEmbeddingsAndPositions = (gradInput: number[][], tokens: number[], learningRate: number, embeddingMatrix: number[][], positionalEncoding: number[][]) => {
  // Обновляем эмбеддинги токенов
  for (let pos = 0; pos < tokens.length; pos++) {
    const tokenId = tokens[pos];
    for (let dim = 0; dim < config.embeddingSize; dim++) {
      embeddingMatrix[tokenId][dim] -= learningRate * gradInput[pos][dim];
    }
  }

  // Обновляем позиционные кодировки
  for (let pos = 0; pos < tokens.length; pos++) {
    for (let dim = 0; dim < config.embeddingSize; dim++) {
      positionalEncoding[pos][dim] -= learningRate * gradInput[pos][dim];
    }
  }
}