import os
import google.genai as genai

# Gemini API setup
GEMINI_MODEL = "gemini-2.0-flash-exp"
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Vocabulary from deaf speech test set (extracted manually from JSON)
vocabulary = [
    "दैनंदिन", "कामे", "दूध", "किती", "आहे", "एक", "लिटर", "द्या", "चाळीस", "रुपयांत", "काय", "मिळेल", "पेपर", "कधी", "येईल", "बस", "गाडी", "कोणत्या", "प्लॅटफॉर्मवर", "मुलुंडला", "कुठे", "उतरायचं", "चालीमूल्य", "मला", "निळे", "जीन्स", "दाखवा", "हे", "दोन", "पॅकेट", "आजचा", "पुढचा", "स्टॉप", "कोणता", "टिकीट", "किंमत", "कमी", "करा", "पन्नास", "रुपये", "देऊ", "शकतो", "चहा", "पाणी", "बिल", "झालं", "हवं", "नाही", "ठीक", "थोडं", "वाट", "बघा", "लवकर", "बाथरूम", "स्टेशन", "इकडेच", "का", "विक्रोली", "वाजता", "पोहोचेल", "सीएसटी", "ला", "जाणारी", "कोणती", "ही", "थानेला", "जाते", "बदलला", "दादरला", "थांबते", "छोटा", "सायझ", "मोठा", "मोजून", "बघू", "रंग", "बदलू", "आकार", "चेक", "फूट", "पाईप", "स्क्रू", "की", "आयरनिंग", "कशी", "करायची", "सांगा", "उद्या", "धुवून", "फोल्ड", "करून", "इकडे", "जातो", "रस्ता", "बरोबर", "वाजले", "उशीर", "झाला", "थांबा", "रोख", "घेतात", "कार्ड", "चालतो", "बाकी", "भाजी", "काळी", "पिशवी", "पांढरी", "ठेवा", "उशीरा", "आला", "तीन", "लोकल", "फास्ट", "लाईन", "स्टॅंड", "ऑटो", "थांबतो", "इथे", "हॉस्पिटल", "पोलीस", "फार्मसी", "टॅब्लेट", "दवा", "डॉक्टर", "अपॉइंटमेंट", "माझं", "नाव", "लिहा", "फोन", "नंबर", "एड्रेस", "सिग्नेचर", "हा", "पैसा", "चिल्लर", "नोट", "बदला", "या", "खोटं", "शॉर्टकट", "हॉटेल", "रुम", "खाली", "रात्री", "रहायचं", "चेकआउट", "लगेच", "येतो", "आता", "मिनिट", "चुकीचं", "पुन्हा", "समजलं", "नीट", "लिहून", "मोबाइलवर", "नकाशा", "जवळ", "एटीएम", "बँक", "पोस्ट", "ऑफिस", "पार्सल", "पाठवायचं", "रेजिस्टर्ड", "कापसाचे", "कपडे", "सिल्कचं", "फिट", "होईल", "ट्राय", "करू", "रिटर्न", "एक्सचेंज", "गिफ्ट", "रॅप", "बॅग", "प्लॅस्टिकची", "कागदाची"
]

print(f"Total words in vocabulary: {len(vocabulary)}")

# Prompt for Gemini to identify high-frequency Marathi words
IDENTIFY_HIGH_FREQ_PROMPT = f"""
You are a Marathi language expert. I have a list of {len(vocabulary)} words extracted from a Marathi speech recognition test set containing everyday transactional sentences (shopping, transportation, banking, medical, etc.).

Your task is to identify which words are HIGH FREQUENCY and commonly used in everyday Marathi conversation. Focus on words that appear frequently in spoken Marathi across different contexts.

From this vocabulary list, select words that meet these criteria:
1. Commonly used in everyday Marathi speech
2. High frequency in conversational Marathi
3. Essential for basic communication
4. Not rare or domain-specific terms

Return your response as a JSON object with:
- "high_frequency_words": array of words that are high frequency in Marathi
- "explanation": brief explanation of your selection criteria

Vocabulary to analyze:
{', '.join(vocabulary)}

Respond only with valid JSON.
"""

print("Asking Gemini to identify high-frequency Marathi words...")

try:
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=IDENTIFY_HIGH_FREQ_PROMPT
    )

    result = response.text.strip()
    print("Gemini response:")
    print(result)

    # Save result
    with open('high_frequency_marathi_words.json', 'w', encoding='utf-8') as f:
        f.write(result)

except Exception as e:
    print(f"Error calling Gemini: {e}")