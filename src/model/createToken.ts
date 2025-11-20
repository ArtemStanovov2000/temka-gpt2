import { textPreparation } from "./textPreparation/textPreparation"

import { gamma1layerFirst } from "../data/matrix/layer_1/gamma1layerFirst"
import { beta1layerFirst } from "../data/matrix/layer_1/beta1layerSecond"
import { gamma1layerSecond } from "../data/matrix/layer_1/gamma1layerSecond"
import { beta1layerSecond } from "../data/matrix/layer_1/beta1layerFirst"
import { W_Q_1layer } from "../data/matrix/layer_1/W_Q_1layer"
import { W_K_1layer } from "../data/matrix/layer_1/W_K_1layer"
import { W_V_1layer } from "../data/matrix/layer_1/W_V_1layer"
import { W_o_1layer } from "../data/matrix/layer_1/W_o_1layer"
import { Weights1_1layer } from "../data/matrix/layer_1/Weights1_1layer"
import { Weights2_1layer } from "../data/matrix/layer_1/Weights2_1layer"
import { biasHidden1layer } from "../data/matrix/layer_1/biasHidden1layer"
import { biasOutput1layer } from "../data/matrix/layer_1/biasOutput1layer"

import { gamma2layerFirst } from "../data/matrix/layer_2/gamma2layerFirst"
import { beta2layerFirst } from "../data/matrix/layer_2/beta2layerFirst"
import { gamma2layerSecond } from "../data/matrix/layer_2/gamma2layerSecond"
import { beta2layerSecond } from "../data/matrix/layer_2/beta2layerSecond"
import { W_Q_2layer } from "../data/matrix/layer_2/W_Q_2layer"
import { W_K_2layer } from "../data/matrix/layer_2/W_K_2layer"
import { W_V_2layer } from "../data/matrix/layer_2/W_V_2layer"
import { W_o_2layer } from "../data/matrix/layer_2/W_o_2layer"
import { Weights1_2layer } from "../data/matrix/layer_2/Weights1_2layer"
import { Weights2_2layer } from "../data/matrix/layer_2/Weights2_2layer"
import { biasHidden2layer } from "../data/matrix/layer_2/biasHidden2layer"
import { biasOutput2layer } from "../data/matrix/layer_2/biasOutput2layer"

import { linearLayer } from "../data/matrix/linearLayer"
import { attention } from "./attention/attention"
import { FFN } from "./FFN/FFN"
import { predictNextToken } from "./linearLayer/predictNextToken"

export const createToken = (text: string) => {
    const initialEmb = textPreparation(text)
    const att1 = attention(initialEmb.embeddings, gamma1layerFirst, beta1layerFirst, gamma1layerSecond, beta1layerSecond, W_Q_1layer, W_K_1layer, W_V_1layer, W_o_1layer)
    const FFN1 = FFN(att1, Weights1_1layer, Weights2_1layer, biasHidden1layer, biasOutput1layer)
    const att2 = attention(FFN1, gamma2layerFirst, beta2layerFirst, gamma2layerSecond, beta2layerSecond, W_Q_2layer, W_K_2layer, W_V_2layer, W_o_2layer)
    const FFN2 = FFN(att2, Weights1_2layer, Weights2_2layer, biasHidden2layer, biasOutput2layer)
    const currentToken = predictNextToken(FFN2, initialEmb.length, linearLayer)
    return currentToken
}