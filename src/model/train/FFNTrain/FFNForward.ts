import { matrixMultiply, addBias, applyActivation, gelu } from "../../FFN/FFN";

export const FFNForward = (
    input: number[][],
    weights1: number[][],
    weights2: number[][],
    biasHidden: number[],
    biasOutput: number[]
): { output: number[][], cache: any } => {
    // Сохраняем вход для обратного прохода
    const cache: any = { input };
    
    // Первое линейное преобразование
    const hidden = matrixMultiply(input, weights1);
    cache.hidden = hidden;
    
    // Добавление смещения
    const hiddenWithBias = addBias(hidden, biasHidden);
    cache.hiddenWithBias = hiddenWithBias;
    
    // Применение активации GELU
    const activated = applyActivation(hiddenWithBias, gelu);
    cache.activated = activated;
    
    // Второе линейное преобразование
    const output = matrixMultiply(activated, weights2);
    cache.outputBeforeBias = output;
    
    // Добавление выходного смещения
    const outputWithBias = addBias(output, biasOutput);
    
    return { output: outputWithBias, cache };
};