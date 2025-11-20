import { embeddingMatrix } from "../../../data/matrix/embeddingMatrix";
import { config } from "../../../utils/config/config";
import { positionMatrix } from "../../../data/matrix/positionMatrix";

export const getTrainingExample = (tokens: number[],) => {
    const contextLength = 64;
    const totalTokens = tokens.length;

    // Максимальный стартовый индекс
    const maxStartIndex = totalTokens - contextLength;

    // Случайный стартовый индекс
    const startIndex = Math.floor(Math.random() * (maxStartIndex + 1));
    
    // Возвращаем 64 токена: [x₀, x₁, ..., x₆₃]
    const exampleArray = tokens.slice(startIndex, startIndex + contextLength)

    // Создаем матрицу эмбеддингов
    const embeddings: number[][] = []
    for (let i = 0; i < exampleArray.length; i++) {
        embeddings.push(embeddingMatrix[i])
    }

    const calculatePosition = (tokens: number[][]) => {
        const poz: number[][] = new Array(config.contextLength)
        for (let i = 0; i < poz.length; i++) {
            poz[i] = new Array(config.embeddingSize)
            for (let j = 0; j < poz[0].length; j++) {
                poz[i][j] = tokens[i][j] + positionMatrix[i][j]
            }
        }
        return poz
    }

    return {embeddingsVector: calculatePosition(embeddings), embeddinsIndex: exampleArray}
}
