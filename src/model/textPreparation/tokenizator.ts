import { vocab } from "../../utils/config/vocab";

//Универсальный токенизатор текста
export const tokenizationText = (text: string) => {
  // Нормализация: пробелы -> _, всё в нижний регистр
    const normalized = text.replace(/\s/g, '_').toLowerCase();
    
    // Преобразование символов в токены
    const tokens = [];
    for (let i = 0; i < normalized.length; i++) {
        const char = normalized[i];
        const tokenId = vocab.indexOf(char);
        
        // Добавляем только символы из словаря
        if (tokenId !== -1) {
            tokens.push(tokenId);
        }
        // Символы вне словаря игнорируем
    }
    
    return tokens;
}



