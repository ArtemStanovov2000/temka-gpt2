export const clipGradients = (grad: number[][], maxNorm: number): number[][] => {
    const norm = Math.sqrt(grad.flat().reduce((sum, val) => sum + val * val, 0));
    if (norm > maxNorm) {
        const scale = maxNorm / norm;
        return grad.map(row => row.map(val => val * scale));
    }
    return grad;
}