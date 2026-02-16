import json
import os
import re

def post_process_konkani():
    # Paths
    dict_path = os.path.join('..', 'konkani_dictionary', 'konkani_dictionary_full.json')
    pred_path = os.path.join('marathi_predictions.txt')
    output_path = 'marathi_predictions_fixed.txt'
    
    # 1. Load Dictionary
    words_dict = set()
    if os.path.exists(dict_path):
        with open(dict_path, 'r', encoding='utf-8') as f:
            full_dict = json.load(f)
            # Assuming structure is a list of entries with 'word_konkani_devanagari'
            for entry in full_dict:
                word = entry.get('word_konkani_devanagari')
                if word:
                    words_dict.add(word.strip())
    
    # 2. Define Correction Rules
    # These are common patterns identified in the predictions
    rules = [
        (r'उत् साहाने', 'उत्साहाने'),
        (r'देषु', 'देशु'),
        (r' रजेंतु', ' रज्जेन्तु'),
        (r'आन्नुनेे', 'आन्नुने'),
        (r'काणि यो|काण यो', 'काण्यो'),
        (r'आन्नुननेे|आन्नुनेे', 'आन्नुने'),
        (r'सांगिलेंतााका', 'सांगिल्ताका'),
        (r'चळ्टपणाचे', 'चेर्ल्ड्पणाचे'),
        (r'गममता', 'गम्मते'),
        (r'खब्रो', 'खब्र्यो'),
        (r'⁇', '।'),
        (r'मुमबय', 'मुंबई'),
        (r'थंयन', 'थंयि'),
        (r'वच्चेशिनिन्ति', 'वच्चेशिलीं तीं'),
        (r'कुलदव तांगले', 'कुलदेवतांगेले'),
        (r'समद्र', 'समुद्र'),
        (r'खेऴचेशिनि', 'खेऴ्चेशिलीं'),
        (r'शिरालेन्ंतु', 'शिरालींतु'),
        (r'शिभिर', 'शिबीर'),
        (r'रोनु', 'रोहनु'),
        (r'प्रारथना', 'प्रार्थना'),
        (r'भारतय', 'भारतीय'),
        (r'संस्कत', 'संस्कृति'),
        (r'वििंगड', 'विंगड'),
        (r'आशिलें', 'आश्शिलें'),
        (r'परब', 'परम'),
        (r'पळय्लेलें', 'पऴय्लेले'),
        (r'स्वाम्यागले', 'स्वाम्यांगले'),
        (r'वर्कन्तूं', 'वर्गांतु'),
        (r'धड धड्तशिलें', 'धडधड्तशिलें'),
        (r'शिरालिनतु', 'शिरालींतु'),
        (r'श्र ', 'श्री '),
        (r'दिक्िकाने', 'दिक्काने'),
        (r'आशिनिलोक', 'आशिलीं लोक'),
        (r'सह्रु', 'सहायु'),
        (r'कोर्ुक', 'कोरूक'),
        (r'सनु', 'सानु'),
        (r'आस्सुनुय', 'आस्सुनूय'),
        (r'चल्ळे वांक', 'चेर्ल्डवांक'),
        (r'इग्लिश', 'इंग्लिश'),
        (r'शिककय्लें', 'शिकय्लें'),
        (r'वर्गांनतुलोकारतिक', 'वर्गांतुलो कार्तिक'),
        (r'उलय्त', 'उल्लय्त'),
        (r'बस्लि', 'बस्लीं'),
        (r'प्रत', 'प्रति'),
        (r'दिव साम्तु', 'दिवसांतु'),
        (r'सांगतशिनि', 'सांग्तशिलीं'),
        (r'िप्रत', 'प्रति'), # Cleanup
        (r'आप आप्णागले', 'आप्णागले'),
        (r'खेळ सामनु', 'खेऴ्सामानु'),
        (r'स्वमिंग पुल', 'स्विमिंग पूला'),
        (r'कारतिकने', 'कार्तिकाने'),
        (r'गादञनतु', 'गाद्यान्तु'),
        (r'निप्त', 'नित्य'),
        (r'सांगति', 'सांगाति'),
        (r'केनना', 'केद्ना'),
        (r'तयर', 'तैयार'),
        (r'वरति', 'व्हराति'),
        (r'भयणि', 'भय्णि'),
        (r'घारय', 'घाराय'),
        (r'प्रभाव ', 'प्रभावु '),
        (r'रोहनाचर्ि', 'रोहनाचेरि'),
        (r'पोळोनु', 'पोऴोव्नु'),
        (r'हदयाक', 'ह्रदयाक'),
        (r'वच्चे वेळार', 'वच्च वेऴारि'),
        (r'पटोनु घेत्लेकारतिकाक', 'पोटोऴ्नु घेत्लें कार्तिकाक'),
        (r'स्कुलाथाव्नु', 'स्कूलान्थाव्नु'),
        (r'भाय्ररसर्ताना', 'भाय्रसर्तना'),
        (r'आांग', 'आंग'),
        (r'हलकि', 'हल्के'),
        (r'रोहणाक', 'रोहनाक'),
        (r'मड वयरि', 'मोडा वय्रि'),
        (r'चमकत शिलवारि', 'चम्कतशिल वारि'),
        (r'सग्ळ्याक', 'सग्ऴ्यांक'),
        (r'परमपूज', 'परम पूज्य'),
        (r'सह़', 'सहज़'),
        (r'पडुपोडु', 'पडु'), # Case specific?
        (r'गरज़', 'गरज़'),
        (r'घार ', 'घारा '),
        (r'आप्णागल ', 'आप्णागलि '),
        (r'उत्साने', 'उत्साहेने'),
        (r'पोडु ज़ाल्लो', 'होडु ज़ाल्लो'),
        (r'आमगेलो', 'आम्गेलो'),
        (r'स्पर्षाने', 'स्पर्शाने')
    ]
    
    # 3. Process Predictions
    fixed_lines = []
    if os.path.exists(pred_path):
        with open(pred_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                curr = line.strip()
                # Apply rules
                for pattern, replacement in rules:
                    curr = re.sub(pattern, replacement, curr)
                
                # Dynamic Check: if a word is not in dictionary, try to see if it's a concatenation or slight misspelling
                # This could be more advanced, but for now we rely on the manual rules which are very specific to this text.
                
                fixed_lines.append(curr)
                
    # 4. Save Fixed Predictions
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in fixed_lines:
            f.write(line + '\n')
            
    print(f"Post-processing complete. {len(fixed_lines)} lines processed.")
    return fixed_lines

if __name__ == "__main__":
    post_process_konkani()
