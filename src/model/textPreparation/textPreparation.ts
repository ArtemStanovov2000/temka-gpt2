import { embeddingMatrix } from "../../data/matrix/embeddingMatrix"
import { tokenizationText } from "./tokenizator"
import { calculatePosition } from "./calculatePosition"

export const textPreparation = (text: string) => {
    // Токенизируем текст
    const tokens = tokenizationText(text)

    // Создаем матрицу эмбеддингов
    const embeddings: number[][] = []
    for (let i = 0; i < tokens.length; i++) {
        embeddings.push(embeddingMatrix[i])
    }

    // Заполняем матрицу контекста токенами-паддингами (длина контекста 512, токенов текста 121, токенов паддингов 512-121 = 391)
    for (let i = tokens.length; i < 64; i++) {
        embeddings.push(new Array(10).fill(0))
    }
    
    // Сложение эмбеддингов с матрицей позиций
    const positionEmb = calculatePosition(embeddings)
    
    return {embeddings: positionEmb, length: tokens.length}
}
