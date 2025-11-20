export const clipGradientsVector = (grad: number[], maxNorm: number): number[] => {
    const norm = Math.sqrt(grad.reduce((sum, val) => sum + val * val, 0));
    if (norm > maxNorm) {
        const scale = maxNorm / norm;
        return grad.map(val => val * scale);
    }
    return grad;
}