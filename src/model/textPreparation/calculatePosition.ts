import { positionMatrix } from "../../data/matrix/positionMatrix"
import { config } from "../../utils/config/config"

// Складывает матрицу токенов с матрицей эмбеддингов, нужна только на 1м шаге
export const calculatePosition = (tokens: number[][]) => {
    const poz: number[][] = new Array(config.contextLength)
    for (let i = 0; i < poz.length; i++) {
        poz[i] = new Array(config.embeddingSize)
        for (let j = 0; j < poz[0].length; j++) {
            poz[i][j] = tokens[i][j] + positionMatrix[i][j]
        }
    }
    return poz
}