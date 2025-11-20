const listEnding = ["_противо"]

const reverseArray = (array: string[]) => {
    const reverseArray = []
    for (let i = 0; i < array.length; i++) {
        reverseArray.push(array[i].split("").reverse().join(""))
    }
    return reverseArray
}

// Убирает подстроку с конца строки в массиве строк
export const cleanEnding = (array: string[]) => {
    const reversingArray: string[] = reverseArray(array)
    const listEndingReverse: string[] = reverseArray(listEnding)
    const cleaningArray: string[] = []
    for (let i = 0; i < array.length; i++) {
        let flag = 0
        for (let j = 0; j < listEndingReverse.length; j++) {
            if (reversingArray[i].startsWith(listEndingReverse[j]) && flag === 0) {
                flag = 1
                cleaningArray.push(reversingArray[i].slice(listEndingReverse[j].length))
            } else {
                continue
            }
        }
        if (flag === 0) {
            cleaningArray.push(reversingArray[i])
        }
    }
    return reverseArray(cleaningArray)
}

// Убирает подстроку с начала строки в массиве строк
export const cleanPrefix = (array: string[]) => {
    const reversingArray: string[] = array
    const listEndingReverse: string[] = listEnding
    const cleaningArray: string[] = []
    for (let i = 0; i < array.length; i++) {
        let flag = 0
        for (let j = 0; j < listEndingReverse.length; j++) {
            if (reversingArray[i].startsWith(listEndingReverse[j]) && flag === 0) {
                flag = 1
                cleaningArray.push(reversingArray[i].slice(listEndingReverse[j].length))
            } else {
                continue
            }
        }
        if (flag === 0) {
            cleaningArray.push(reversingArray[i])
        }
    }
    return cleaningArray
}
