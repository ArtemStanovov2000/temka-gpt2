// Разделяет текст на слова
export const splitText = (text: string) => {
    const textArr = text.split("_")
    const valueTrueOnly = []
    for (let i = 0; i < textArr.length; i++) {
        if(textArr[i] !== "") {
            valueTrueOnly.push(textArr[i])
        }
    }
    return valueTrueOnly
}