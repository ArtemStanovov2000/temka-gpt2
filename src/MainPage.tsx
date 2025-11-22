import { useEffect, useState, type FC } from "react"
import { createUseStyles } from "react-jss"
import { generateNextToken } from "./model/createToken";
import { useDispatch, useSelector } from "react-redux";
import { addContext } from "./store/contextSlice";
import { train } from "./model/train";
import { sample } from "./data/educationalSample/sample";
import { miniSample } from "./data/educationalSample/miniSample";
import { embeddingMatrix } from "./data/matrix/embeddingMatrix";
import { positionMatrix } from "./data/matrix/positionMatrix";
import { beta1layerFirst } from "./data/matrix/layer_1/beta1layerFirst";
import { beta1layerSecond } from "./data/matrix/layer_1/beta1layerSecond";
import { biasHidden1layer } from "./data/matrix/layer_1/biasHidden1layer";
import { biasOutput1layer } from "./data/matrix/layer_1/biasOutput1layer";
import { gamma1layerFirst } from "./data/matrix/layer_1/gamma1layerFirst";
import { gamma1layerSecond } from "./data/matrix/layer_1/gamma1layerSecond";
import { W_K_1layer } from "./data/matrix/layer_1/W_K_1layer";
import { W_o_1layer } from "./data/matrix/layer_1/W_o_1layer";
import { W_Q_1layer } from "./data/matrix/layer_1/W_Q_1layer";
import { W_V_1layer } from "./data/matrix/layer_1/W_V_1layer";
import { Weights1_1layer } from "./data/matrix/layer_1/Weights1_1layer";
import { Weights2_1layer } from "./data/matrix/layer_1/Weights2_1layer";
import { beta2layerFirst } from "./data/matrix/layer_2/beta2layerFirst";
import { beta2layerSecond } from "./data/matrix/layer_2/beta2layerSecond";
import { biasHidden2layer } from "./data/matrix/layer_2/biasHidden2layer";
import { biasOutput2layer } from "./data/matrix/layer_2/biasOutput2layer";
import { gamma2layerFirst } from "./data/matrix/layer_2/gamma2layerFirst";
import { gamma2layerSecond } from "./data/matrix/layer_2/gamma2layerSecond";
import { W_K_2layer } from "./data/matrix/layer_2/W_K_2layer";
import { W_o_2layer } from "./data/matrix/layer_2/W_o_2layer";
import { W_Q_2layer } from "./data/matrix/layer_2/W_Q_2layer";
import { W_V_2layer } from "./data/matrix/layer_2/W_V_2layer";
import { Weights1_2layer } from "./data/matrix/layer_2/Weights1_2layer";
import { Weights2_2layer } from "./data/matrix/layer_2/Weights2_2layer";
import { linearLayer } from "./data/matrix/linearLayer";

const useStyles = createUseStyles({
    page: {
        backgroundColor: "#1E1E1E",
        width: "100%",
        height: "100vh",
        display: "flex",
        flexDirection: "column",
    },
    inputBox: {
        display: "flex",
        alignItems: "center",
    },
    input: {
        fontSize: "20px",
        border: "none",
        width: "80%",
        height: "50px",
        margin: "20px",
        paddingLeft: "20px",
        backgroundColor: "#2D2D2D",
        color: "#FFFFFF",
        borderRadius: "4px",
    },
    button: {
        fontSize: "20px",
        height: "52px",
        border: "none",
        padding: "0 20px",
        backgroundColor: "#909090",
        color: "#1E1E1E",
        borderRadius: "4px",
        margin: "0 5px",
        "&:hover": {
            cursor: "pointer",
            backgroundColor: "#d4d4d4",
        },
        "&:active": {
            cursor: "pointer",
            backgroundColor: "#ffffff",
        },
        "&:disabled": {
            backgroundColor: "#555555",
            color: "#888888",
            cursor: "not-allowed",
        }
    },
    output: {
        color: "#FFFFFF",
        padding: "20px",
        fontSize: "18px",
        whiteSpace: "pre-wrap",
        overflowY: "auto",
        flexGrow: 1,
    },
    status: {
        color: "#FFFFFF",
        padding: "10px 20px",
        fontSize: "16px",
    },
    buttonContainer: {
        display: "flex",
        justifyContent: "center",
        padding: "10px",
    }
});

// Функция для скачивания матриц в виде JSON-файла
function downloadMatrices() {
    const matrices = {
        embeddingMatrix,
        positionMatrix,
        beta1layerFirst,
        beta1layerSecond,
        biasHidden1layer,
        biasOutput1layer,
        gamma1layerFirst,
        gamma1layerSecond,
        W_K_1layer,
        W_o_1layer,
        W_Q_1layer,
        W_V_1layer,
        Weights1_1layer,
        Weights2_1layer,

        beta2layerFirst,
        beta2layerSecond,
        biasHidden2layer,
        biasOutput2layer,
        gamma2layerFirst,
        gamma2layerSecond,
        W_K_2layer,
        W_o_2layer,
        W_Q_2layer,
        W_V_2layer,
        Weights1_2layer,
        Weights2_2layer,

        linearLayer
    };

    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(matrices, null, 2));
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", "matrices.json");
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
}

export const MainPage: FC = () => {
    const classes = useStyles()
    const [text, setText] = useState<string>("")
    const [isTraining, setIsTraining] = useState<boolean>(false)
    const [trainingProgress, setTrainingProgress] = useState<number>(0)
    const [trainingTotal, setTrainingTotal] = useState<number>(0)
    const dispatch = useDispatch()
    const context: string = useSelector((store: any) => store.context.context)

    // Запуск createToken при изменении контекста
    useEffect(() => {
        const timer = setTimeout(() => {
            if (context.length > 0) {
                dispatch(addContext(generateNextToken(context.toLocaleLowerCase())))
            }
        }, 100);
        return () => clearTimeout(timer);
    }, [context])

    const inputText = (e: React.ChangeEvent<HTMLInputElement>) => {
        setText(e.target.value)
    }

    const postText = () => {
        if (text.trim()) {
            dispatch(addContext(text))
            setText("")
        }
    }

    const cleanText = () => {
        dispatch(addContext(""))
    }

    // Функция для запуска обучения с пакетной обработкой

    const startTraining = () => {
        setIsTraining(true)
        setTrainingProgress(0)
        setTrainingTotal(300)

        const total = 10000;
        const batchSize = 10000;
        const delay = 100;

        let executed = 10;

        function runNextBatch() {
            // Выполняем пачку вызовов
            for (let i = 0; i < batchSize && executed < total; i++) {
                executed++;
                console.log(executed);
                train(miniSample, 0.00002)
                setTrainingProgress(executed)
            }

            // Если еще не все вызовы выполнены - планируем следующую пачку
            if (executed < total) {
                setTimeout(runNextBatch, delay);
            } else {
                console.log("Все вызовы завершены!");
                setIsTraining(false)

                // Автоматически скачиваем матрицы после завершения
                //downloadMatrices();
            }
        }

        // Запускаем первый пакет
        runNextBatch();
    }



    return (
        <div className={classes.page}>
            <div className={classes.inputBox}>
                <input
                    onInput={inputText}
                    value={text}
                    placeholder="Введите текст..."
                    className={classes.input}
                />
                <button onClick={postText} className={classes.button}>Ввод</button>
            </div>
            <div className={classes.output}>
                <div><strong>Контекст:</strong> {context}</div>
            </div>
            <div className={classes.status}>
                {isTraining ? (
                    <div>Обучение: {trainingProgress} / {trainingTotal}</div>
                ) : (
                    <div>Готово к обучению</div>
                )}
            </div>
            <div className={classes.buttonContainer}>
                <button onClick={cleanText} className={classes.button} disabled={isTraining}>
                    Очистить контекст
                </button>
                <button onClick={startTraining} className={classes.button} disabled={isTraining}>
                    {isTraining ? 'Обучение...' : 'Начать обучение'}
                </button>
                <button onClick={downloadMatrices} className={classes.button} disabled={isTraining}>
                    Скачать матрицы
                </button>
            </div>
        </div>
    )
}

export default MainPage