import { updateWeights, softmaxCrossEntropyBackward, linearBackward, linearForward, calculateLoss } from "./linearLayerTrain";

export const updateLinearLayer = (linearLayer: number[][], FFN2: number[][], tokens: number[], learningRate: number) => {
    // Прямой проход через линейный слой
    const logits = linearForward(FFN2, linearLayer);

    // Вычисление потерь
    const loss = calculateLoss(logits, tokens);

    // Обратное распространение
    const gradOutput = softmaxCrossEntropyBackward(logits, tokens);
    const { gradWeights, gradInput } = linearBackward(FFN2, gradOutput, linearLayer);

    // Обновление весов линейного слоя
    updateWeights(linearLayer, gradWeights, learningRate);
    return {gradInput, loss}
}