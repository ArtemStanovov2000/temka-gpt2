import { embeddingMatrix } from "../../data/matrix/embeddingMatrix"
import { positionMatrix } from "../../data/matrix/positionMatrix"

import { gamma1layerFirst } from "../../data/matrix/layer_1/gamma1layerFirst"
import { beta1layerFirst } from "../../data/matrix/layer_1/beta1layerSecond"
import { gamma1layerSecond } from "../../data/matrix/layer_1/gamma1layerSecond"
import { beta1layerSecond } from "../../data/matrix/layer_1/beta1layerFirst"
import { W_Q_1layer } from "../../data/matrix/layer_1/W_Q_1layer"
import { W_K_1layer } from "../../data/matrix/layer_1/W_K_1layer"
import { W_V_1layer } from "../../data/matrix/layer_1/W_V_1layer"
import { W_o_1layer } from "../../data/matrix/layer_1/W_o_1layer"
import { Weights1_1layer } from "../../data/matrix/layer_1/Weights1_1layer"
import { Weights2_1layer } from "../../data/matrix/layer_1/Weights2_1layer"
import { biasHidden1layer } from "../../data/matrix/layer_1/biasHidden1layer"
import { biasOutput1layer } from "../../data/matrix/layer_1/biasOutput1layer"

import { gamma2layerFirst } from "../../data/matrix/layer_2/gamma2layerFirst"
import { beta2layerFirst } from "../../data/matrix/layer_2/beta2layerFirst"
import { gamma2layerSecond } from "../../data/matrix/layer_2/gamma2layerSecond"
import { beta2layerSecond } from "../../data/matrix/layer_2/beta2layerSecond"
import { W_Q_2layer } from "../../data/matrix/layer_2/W_Q_2layer"
import { W_K_2layer } from "../../data/matrix/layer_2/W_K_2layer"
import { W_V_2layer } from "../../data/matrix/layer_2/W_V_2layer"
import { W_o_2layer } from "../../data/matrix/layer_2/W_o_2layer"
import { Weights1_2layer } from "../../data/matrix/layer_2/Weights1_2layer"
import { Weights2_2layer } from "../../data/matrix/layer_2/Weights2_2layer"
import { biasHidden2layer } from "../../data/matrix/layer_2/biasHidden2layer"
import { biasOutput2layer } from "../../data/matrix/layer_2/biasOutput2layer"

import { linearLayer } from "../../data/matrix/linearLayer"

import { FFNForward } from "./FFNTrain/FFNForward"
import { getTrainingExample } from "./getTrainingExample/getTrainingExample"
import { updateLinearLayer } from "./linearLayerTrain/updateLinearLayer"
import { attentionForward } from "./attentionTrain/attentionForward"
import { FFNBackward, updateFFNWeights } from "./FFNTrain/FFNBackward"
import { attentionBackward } from "./attentionTrain/attentionBackward"
import { updateAttentionWeights } from "./attentionTrain/updateAttentionWeights"
import { updateEmbeddingsAndPositions } from "./updateEmbeddingsAndPositions/updateEmbeddingsAndPositions"


export const train = (tokens: number[], learningRate: number = 0.0002) => {
    const initialEmb = getTrainingExample(tokens)
    const { output: att1, cache: att1Cache } = attentionForward(initialEmb.embeddingsVector, gamma1layerFirst, beta1layerFirst, gamma1layerSecond, beta1layerSecond, W_Q_1layer, W_K_1layer, W_V_1layer, W_o_1layer)
    const { output: FFN1, cache: FFN1Cache } = FFNForward(att1, Weights1_1layer, Weights2_1layer, biasHidden1layer, biasOutput1layer)
    const { output: att2, cache: att2Cache } = attentionForward(FFN1, gamma2layerFirst, beta2layerFirst, gamma2layerSecond, beta2layerSecond, W_Q_2layer, W_K_2layer, W_V_2layer, W_o_2layer)
    const { output: FFN2, cache: FFN2Cache } = FFNForward(att2, Weights1_2layer, Weights2_2layer, biasHidden2layer, biasOutput2layer)

    const { gradInput, loss } = updateLinearLayer(linearLayer, FFN2, initialEmb.embeddinsIndex, learningRate)
    const { dInput: dFFN2Input, dW1: dW1_2, db1: db1_2, dW2: dW2_2, db2: db2_2 } = FFNBackward(gradInput, FFN2Cache, Weights1_2layer, Weights2_2layer, learningRate);
    updateFFNWeights(Weights1_2layer, Weights2_2layer, biasHidden2layer, biasOutput2layer, dW1_2, dW2_2, db1_2, db2_2, learningRate);
    const { dInput: dAtt2Input, dGamma1: dGamma2_1, dBeta1: dBeta2_1, dGamma2: dGamma2_2, dBeta2: dBeta2_2, dW_Q: dW_Q_2, dW_K: dW_K_2, dW_V: dW_V_2, dW_O: dW_O_2 } = attentionBackward(dFFN2Input, att2Cache, learningRate);
    updateAttentionWeights(W_Q_2layer, W_K_2layer, W_V_2layer, W_o_2layer, gamma2layerFirst, beta2layerFirst, gamma2layerSecond, beta2layerSecond, dW_Q_2, dW_K_2, dW_V_2, dW_O_2, dGamma2_1, dBeta2_1, dGamma2_2, dBeta2_2, learningRate);
    const { dInput: dFFN1Input, dW1: dW1_1, db1: db1_1, dW2: dW2_1, db2: db2_1 } = FFNBackward(dAtt2Input, FFN1Cache, Weights1_1layer, Weights2_1layer, learningRate);
    updateFFNWeights(Weights1_1layer, Weights2_1layer, biasHidden1layer, biasOutput1layer, dW1_1, dW2_1, db1_1, db2_1, learningRate);
    const { dInput: dAtt1Input, dGamma1: dGamma1_1, dBeta1: dBeta1_1, dGamma2: dGamma1_2, dBeta2: dBeta1_2, dW_Q: dW_Q_1, dW_K: dW_K_1, dW_V: dW_V_1, dW_O: dW_O_1 } = attentionBackward(dFFN1Input, att1Cache, learningRate);
    updateAttentionWeights(W_Q_1layer, W_K_1layer, W_V_1layer, W_o_1layer, gamma1layerFirst, beta1layerFirst, gamma1layerSecond, beta1layerSecond, dW_Q_1, dW_K_1, dW_V_1, dW_O_1, dGamma1_1, dBeta1_1, dGamma1_2, dBeta1_2, learningRate);
    updateEmbeddingsAndPositions(dAtt1Input, initialEmb.embeddinsIndex, learningRate, embeddingMatrix, positionMatrix)
    console.log(loss);
    return loss
}