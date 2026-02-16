const fs = require('fs');
const path = require('path');

const CANDIDATES_FILE = path.join(__dirname, '../story_candidates.json');

// IAST Mapping rules (simplified)
function toIAST(text) {
    if (!text) return '';
    let result = text
        .replace(/अ/g, 'a')
        .replace(/आ/g, 'ā')
        .replace(/इ/g, 'i')
        .replace(/ई/g, 'ī')
        .replace(/उ/g, 'u')
        .replace(/ऊ/g, 'ū')
        .replace(/ऋ/g, 'ṛ')
        .replace(/ए/g, 'e')
        .replace(/ऐ/g, 'ai')
        .replace(/ओ/g, 'o')
        .replace(/औ/g, 'au')
        .replace(/क/g, 'ka')
        .replace(/ख/g, 'kha')
        .replace(/ग/g, 'ga')
        .replace(/घ/g, 'gha')
        .replace(/ङ/g, 'ṅa')
        .replace(/च/g, 'ca')
        .replace(/छ/g, 'cha')
        .replace(/ज/g, 'ja')
        .replace(/झ/g, 'jha')
        .replace(/ञ/g, 'ña')
        .replace(/ट/g, 'ṭa')
        .replace(/ठ/g, 'ṭha')
        .replace(/ड/g, 'ḍa')
        .replace(/ढ/g, 'ḍha')
        .replace(/ण/g, 'ṇa')
        .replace(/त/g, 'ta')
        .replace(/थ/g, 'tha')
        .replace(/द/g, 'da')
        .replace(/ध/g, 'dha')
        .replace(/न/g, 'na')
        .replace(/प/g, 'pa')
        .replace(/फ/g, 'pha')
        .replace(/ब/g, 'ba')
        .replace(/भ/g, 'bha')
        .replace(/म/g, 'ma')
        .replace(/य/g, 'ya')
        .replace(/र/g, 'ra')
        .replace(/ल/g, 'la')
        .replace(/व/g, 'va')
        .replace(/श/g, 'śa')
        .replace(/ष/g, 'ṣa')
        .replace(/स/g, 'sa')
        .replace(/ह/g, 'ha')
        .replace(/ळ/g, 'ḷa')
        .replace(/क्ष/g, 'kṣa')
        .replace(/ज्ञ/g, 'jña')
        .replace(/ा/g, 'ā')
        .replace(/ि/g, 'i')
        .replace(/ी/g, 'ī')
        .replace(/ु/g, 'u')
        .replace(/ू/g, 'ū')
        .replace(/ृ/g, 'ṛ')
        .replace(/े/g, 'e')
        .replace(/ै/g, 'ai')
        .replace(/ो/g, 'o')
        .replace(/ौ/g, 'au')
        .replace(/ं/g, 'ṃ')
        .replace(/ः/g, 'ḥ')
        .replace(/्/g, '')
        .replace(/़/g, '');
    return result;
}

const inferredMeanings = {
    // --- Story 1 Words ---
    "चल": "walk / move", "रे": "hey", "भोपळा": "pumpkin", "टुनुक": "hop",
    "एकी": "one (fem)", "गोम्टी": "beautiful", "काणी": "story", "आय्कयाति": "listen",
    "घरांतु": "in house", "एक्ऴि": "alone (fem)", "राब्तालि": "lived", "भाय्रचि": "outside",
    "होऴ्ळें": "big", "रान": "forest", "रान्नाचे": "of forest", "आनेक": "many",
    "दिक्काक": "direction", "एकु": "one (masc)", "गांवु": "village", "आशिल्लो": "was",
    "गांवान्तु": "in village", "आज्जेगली": "grandma's", "धूव": "daughter", "तिगल्या": "her",
    "कुटुम्बा": "family", "सा‌ंगात्ति": "with", "दिवसु": "day", "धूवेगल": "daughter's",
    "घारा": "home", "वच्चुक": "to go", "भाय्रसर्लि": "started/left", "सान": "small",
    "चिल्लान्तु": "in bag", "थोडो": "some", "सामानु": "stuff", "घेव्नु": "taking",
    "रानान्तु": "in forest", "चम्कुंचाक": "to walk", "सूरु": "start", "केल्लें": "did",
    "तिन्नें": "she (instr)", "पाव": "quarter/reached", "वाट": "way", "दाण्टुनु": "crossing",
    "वत्ता": "going", "म्हण्तना": "saying", "तिका": "to her", "सिंहु": "lion",
    "मेऴ्ळो": "met", "म्हळालो": "said", "आज्जी": "grandma", "ज़ोरु": "strong/loud",
    "भूक": "hunger", "लाग्ल्या": "felt", "माका": "me", "हांव": "I", "तूक्का": "you",
    "खाव्नु": "eating", "सोड्ता": "leave/eat", "हांग": "here", "यो": "come",
    "बुद्वन्ति": "wise", "म्हात्र": "only", "न्हयि": "no/not", "धैर्यवान": "brave",
    "आश्शिलि": "was (fem)", "घड": "moment", "भित्तरि": "inside", "तिन्ने": "she",
    "सिंहाक": "to lion", "उत्तर": "answer", "दिल्लें": "gave", "कल्लें": "what",
    "तूं": "you", "अय्यो": "alas", "देवा": "god", "मगले": "my", "हात": "hand",
    "पाय": "leg", "सुक्किले": "dry", "बड्यो": "sticks", "श्यो": "like", "आस्सति": "are",
    "पऴे": "see", "ताज्जे": "its", "बद्लाक": "instead", "तीन": "three", "चार": "four",
    "दिवस": "days", "राक्ल्यारि": "if wait", "ज़ाय्शना": "won't happen", "वे": "eh?",
    "धूवेगले": "daughter's", "वत्तस": "going", "थंयि": "there", "जेव्नु": "eating",
    "टवटवी": "fresh/plump", "ज़ाव्नु": "becoming", "येत्त": "will come", "तावळि": "then",
    "खाव्येद": "can eat", "व्हयि": "yes", "म्हुणु": "saying", "दिस्लें": "felt",
    "ज़ाय्द": "okay", "ज़ाल्यारि": "then/if", "वग्गि": "fast", "हेंचि": "this same",
    "घे": "take", "खुशालेरि": "happily", "मुखारि": "forward", "वचुलि": "went",
    "अर्द": "half", "वागु": "tiger", "कोल्लो": "fox", "पाव्ली": "reached", "आम्गेलि": "our",
    "म्हय्नो": "month", "राब्लि": "stayed", "वच्चे": "going", "पय्ले": "first/before",
    "धूवेक": "to daughter", "सिंहागले": "lion's", "वागागले": "tiger's", "कोल्ल्या": "fox's",
    "सांग्ले": "told", "तिगली": "her", "शीदा": "straight", "वचुनु": "going", "होडु": "big",
    "आय्लि": "came", "तांतु": "in that", "थाव्नु": "from", "कुड्को": "piece", "काणु": "taking out",
    "ताजे": "its", "प्हुट": "hole", "कोर्नु": "doing", "आज्जे": "grandma's", "खतिरि": "sake",
    "ज़ागो": "place", "केल्लो": "made", "बस्लि": "sat", "लकय्लो": "roll", "अद्भुत": "wonderful",
    "रान्ना": "forest", "वचुलो": "went", "पद": "song", "म्होणुक": "to sing", "पळय्ल्या": "saw",
    "राक्तस": "waiting", "येनिचि": "coming", "ती": "she", "आवाज़ु": "voice", "सपूर": "thin",
    "ना": "no", "बा": "dear", "कोणि": "who", "माक": "me", "गोत्ना": "dont know", "म": "man",
    "चिक्केचि": "little", "लकय्त": "roll", "मूर्ख": "fool", "लकय्लें": "rolled",
    "कोल्ल्याने": "fox (instr)", "वागाने": "tiger (instr)", "सिंहाने": "lion (instr)",
    "आज्जीगले": "grandma's", "पाव्लो": "reached", "बागिल": "door", "घाल्नु": "putting",
    "घेत्लें": "closed/took", "आवड्ले": "liked", "ताळि": "clap", "वाज़ोयाँ": "play/clap",

    // --- Story 2 Words (New) ---
    "दक्ष": "Daksha",
    "प्रजापतिंगले": "Prajapati's",
    "यज्ञ": "Yagna/Sacrifice",
    "व्यस्त": "busy",
    "आशिलो": "was",
    "सग्ळ्यांतु": "amongst all",
    "श्रेष्ठ": "best/supreme",
    "आयोजित": "organized",
    "कर्तशिलो": "was doing",
    "तैय्यारी": "preparation",
    "कोर्चे": "to do",
    "नयशिले": "was not",
    "काम": "work",
    "ताका": "to him",
    "पूर्ण": "full",
    "विश्वासु": "confidence",
    "कि": "that",
    "तागले": "his",
    "हाताने": "by hands",
    "ज़ात्ले": "will happen",
    "हे": "this",
    "महान": "great",
    "ब्रह्म": "Brahma",
    "विष्णु": "Vishnu",
    "बाकि": "rest/other",
    "सग्ळ": "all",
    "देवांक": "to gods",
    "पेटय्ले": "sent",
    "आमंत्रण": "invitation",
    "सप्त": "seven",
    "ऋषिंक": "to rishis",
    "रित्विक": "priest",
    "कोरूक": "to do",
    "निमंत्रण": "invitation",
    "हवनकुण्ड": "fire pit",
    "विशेष": "special",
    "निर्माण": "construction",
    "करय्लें": "got done",
    "एकि": "one",
    "शुभ": "auspicious",
    "तिथि": "date",
    "ठरोनु": "deciding",
    "अतिथि": "guest",
    "स्वागताचे": "of welcome",
    "आयोजन": "arrangement",
    "ठरय्लें": "decided",
    "इत्ल": "this much",
    "सग्ळें": "everything",
    "ताज़ो": "his",
    "अशुभ": "inauspicious",
    "ध्येयु": "goal",
    "महादेवागले": "Shiva's",
    "अपमान": "insult",
    "कोर्चैं": "doing",
    "एकमात्र": "only one",
    "तागलो": "his",
    "निश्चयु": "decision",
    "आप्णागले": "his own",
    "ज़ांवय": "son-in-law",
    "सतीगले": "Sati's",
    "पति": "husband",
    "आशिलें": "was",
    "ईश्वराक": "to God (Shiva)",
    "आमंत्रित": "invited",
    "करर्नाशि": "without doing",
    "तिरस्कार": "hatred/scorn",
    "कोर्चें": "to do",
    "मनान्तु": "in mind",
    "बश्शिलें": "sat/fixed",
    "ढोलु": "drum",
    "वाज़प": "music",
    "वेदमंत्रांचो": "of vedic mantras",
    "उद्घोषु": "chanting",
    "चार्रि": "all four",
    "दिकाने": "directions",
    "ज़ाल्ले": "happened",
    "वातावरण": "atmosphere",
    "विशेषु": "special",
    "यज्ञाक": "to yagna",
    "वत्तशिले": "going",
    "देवांगलि": "gods'",
    "सवारी": "procession",
    "पळय्ली": "saw",
    "सती": "Sati",
    "ने": "by",
    "अर्रे": "Oh!",
    "कस्लकि": "something",
    "चुक्क्ल्यां": "mistake/missed",
    "कल्याक": "why",
    "सांगनि": "didnt tell",
    "आम्का": "us",
    "अशि": "like this",
    "विचारु": "thought",
    "नाक्का": "dont want",
    "वच्चें": "go",
    "म्हुणु": "saying",
    "ईश्वराने": "Ishwar (Shiva)",
    "प्रयत्न": "effort",
    "केले": "did",
    "सांगुक": "to tell",
    "भाय्रसर्ल": "set out",
    "दक्षागले": "Daksha's",
    "वचुक": "to go",
    "महादेवाक": "to Mahadev",
    "ज्ञान": "knowledge",
    "विकारात्मक": "destructive",
    "हेतृचो": "of intent",
    "कल्पना": "imagination/idea",
    "भयानक": "terrible",
    "भविष्याचो": "of future",
    "तिगलें": "her",
    "मन": "mind",
    "पवित्र": "pure",
    "तिक्का": "to her",
    "कस्ले": "anything",
    "दिस्सनि": "didnt see",
    "विचित्र": "strange",
    "तात": "father",
    "कामाचे": "work's",
    "गोंदोळांतु": "confusion",
    "आम्का": "us",
    "आपोंचाक": "to call",
    "विसर्लो": "forgot",
    "आस्का": "must be",
    "पुणि": "at least",
    "वच्का": "should go",
    "नाज़ाल्यारि": "otherwise",
    "वाय्ट": "bad",
    "दिस्स्तलें": "will feel",
    "नवे": "no?",
    "नंदी": "Nandi",
    "सांगाति": "with",
    "पुत्री": "daughter",
    "दक्षागलि": "Daksha's",
    "वत्ना": "going",
    "दुखि": "sad",
    "महादेव": "Mahadev",
    "पळय्त": "watching",
    "राब्लो": "stood",
    "अर्धांगिणिक": "wife",
    "यज्ञस्थळि": "yagna place",
    "पाव्नु": "reaching",
    "आनन्दाने": "happily",
    "बाप्सुक": "to father",
    "प्रणामु": "bow",
    "विस्मित": "surprised",
    "ज़ाल्लि": "became",
    "दोळे": "eyes",
    "पळय्तशिले": "were looking",
    "भयंकर": "terrible",
    "कोप्पाने": "anger",
    "शुभकार्यान्तु": "in auspicious work",
    "महाविघ्न": "great obstacle",
    "आय्लें": "came",
    "स्वताले": "her own",
    "शरीर": "body",
    "त्याग": "sacrifice",
    "योगबलाने": "by yogic power",
    "पीडित": "pained",
    "ज़ाल्लो": "became",
    "तीव्र": "intense",
    "शोकाने": "grief",
    "वीरभद्र": "Veerbhadra",
    "शिरच्छेदन": "beheading",
    "तान्नें": "he",
    "मागेर्चि": "later",
    "सग्ळ्यांक": "to everyone",
    "गोत्तशिलि": "knew",
    "हिमालय": "Himalaya",
    "परत": "again",
    "अवतरित": "incarnated",
    "बोकड्येगले": "goat's",
    "मात्तें": "head",
    "जीवन": "life",
    "व्यतीत": "spent",
    "श्रेष्ठ": "best",
    "आस्सुनुय": "even being",
    "परिणामु": "result",
    "विपरीत": "opposite",
    "कस्ल्यक": "why",
    "प्रश्नांचे": "questions'",
    "सकारात्मक": "positive",
    "आय्लो": "came",
    "तरि": "then",
    "श्रेष्ठांतु": "amongst best",
    "मेळ्ता": "get",
    "अवशय": "surely",
    "उज्ज्वल": "bright",
    "सफल": "successful",
    "भविष्य": "future",

    // --- Story 6 Words (Crow and Sparrow) ---
    "काय्ळो": "crow",
    "आनी": "and",
    "गुब्ची": "sparrow",
    "ही": "this (fem)",
    "गुब्चीगलि": "sparrow's",
    "ज़ारारि": "under/beneath",
    "ज़ाडारि": "under/beneath",
    "गूडांतु": "in nest",
    "राब्तालो": "lived (masc)",
    "लाग्गि": "near",
    "माडारी": "nest-dweller / one who builds nest",
    "पाव्सु": "rain",
    "काय्ळ्यागलें": "crow's",
    "गूडु": "nest",
    "ज़ारार्थाव्नु": "blew away/destroyed",
    "ज़ाडार्थाव्नु": "blew away/destroyed",
    "तग्गु": "completely",
    "पळ्ळें": "fell",
    "तिंबिलो": "wet",
    "कड्कड्तचि": "cawing",
    "गुब्चीगले": "sparrow's",
    "धाडाय्लें": "knocked",
    "भित्तर्थाव्नु": "from inside",
    "निम्गिले": "came out",
    "कोण": "who",
    "काडी": "open",
    "राब": "wait",
    "मगल्ले": "my",
    "पिल्लांक": "to children/chicks",
    "न्हाणय्तस": "bathing",
    "माग्गेरि": "later",
    "काड्तां": "will open",
    "वेळाने": "by time",
    "लाय्तस": "bringing",
    "निद्कारय्तस": "putting to sleep",
    "अब्बा": "oh!",
    "काळें": "opened",
    "हुस्स": "happy",
    "ज़ार्लें": "became",
    "गुबक्का": "sparrow (vocative)",
    "आजि": "today",
    "रात्रि": "night",
    "राब्बुक": "to stay",
    "दित्त": "give",
    "कोमल": "soft",
    "खंयि": "where",
    "निद्दता": "sleeping",
    "न्हाण्ये": "bath",
    "आस्स्": "is",
    "नाक": "no",
    "वल्ल": "rain",
    "ज़ार्ला": "got wet",
    "रांचवासरेंतु": "in daytime",
    "उज्जान्तु": "by fire",
    "ज़ोळ्नु": "to warm",
    "वचद": "going",
    "कपटि": "cunning",
    "पाळ्यां": "room/chamber",
    "पिल्लांगले": "children's",
    "ओहो": "oh",
    "मूळांतु": "in corner",
    "निदलो": "slept",
    "ज़ारि": "awake",
    "कुटुम": "pecking sound",
    "काकोबा": "crow (vocative)",
    "कस्ल": "what",
    "शब्दु": "sound",
    "तान्ने": "he",
    "कांय": "nothing",
    "चणे": "chickpeas",
    "खांव्चाक": "to eat",
    "दिल्लेलें": "had given",
    "खात्तशिलों": "was eating",
    "भोळीची": "innocent/simple",
    "लेक्लें": "believed",
    "दुस्र": "second",
    "प्हाल्प्हाल्यारि": "early morning",
    "उगळ्ळें": "opened",
    "घड्ये": "pot",
    "उब्बुनु": "swollen/sitting",
    "गेल्लो": "went",
    "हाका": "to her",
    "अंवसर": "misfortune",
    "आस्सो": "alas",
    "लेक्तचि": "thinking",
    "पाळ्ळ्यां": "to rooms",
    "नाशिलीं": "destroyed",
    "रोडुक": "crying",
    "कळें": "understood",
    "क्रूर": "cruel",
    "तांका": "them",
    "सोळेल्लें": "had eaten",
    "देवभक्त": "devout",
    "आशिली": "was (fem)",
    "गोम्टो": "good/clever",
    "उपायु": "plan/solution",
    "काळ्ळो": "found/made",
    "बय्सुनु": "to announce",
    "घोषणा": "announcement",
    "धोन्पारा": "afternoon",
    "तुम्का": "to you all",
    "काय्ळ्यांक": "to crows",
    "जेव्णाक": "to eat",
    "खंडित": "without fail",
    "येयाति": "come",
    "आय्कलें": "heard",
    "तोंडान्तु": "in mouth",
    "उदाक": "saliva",
    "तोवय": "then",
    "राकूनु": "keeping/waiting",
    "सग्ळ्यांखतिरि": "for all",
    "बूंदी": "boondi (sweet)",
    "उंडे": "ladoos",
    "केल्लेलें": "had made",
    "एकेक": "one by one",
    "तोंडांतु": "in mouth",
    "घाल्तशिलि": "was putting",
    "केद्ना": "when",
    "उण्डो": "ladoo",
    "हुन्हुन": "piping hot",
    "सोळ्ळें": "to swallow",
    "दुराशि": "greedy",
    "अन्त": "end",
    "पोटांतु": "in stomach",
    "पणा": "but",
    "प्हुडे": "burst",
    "पोट": "stomach",
    "प्हुट्लें": "burst",
    "चीव": "chirp",
    "कर्तचि": "doing",
    "पोटोल्नु": "to hug/embrace",
    "तिगल्लीं": "her",
    "खुशालेंतु": "happily"
};

async function augmentCandidates() {
    console.log('🧪 Augmenting candidates with AI logic...');

    if (!fs.existsSync(CANDIDATES_FILE)) {
        console.error('❌ Candidates file not found.');
        process.exit(1);
    }

    const candidates = JSON.parse(fs.readFileSync(CANDIDATES_FILE, 'utf8'));

    candidates.forEach(c => {
        // AI-Deduce Meaning
        const cleanWord = c.word.replace(/[।।|!?,."()\-–\[\]]/g, '').trim();
        if (inferredMeanings[cleanWord]) {
            c.ai_meaning = inferredMeanings[cleanWord];
        }

        // Generate IAST
        if (c.action === 'ADD' || (c.original_entry && !c.original_entry.word_konkani_english_alphabet)) {
            let iast = toIAST(cleanWord);

            // Overrides for proper Konkani phonetics
            const overrides = {
                "आज्जी": "ājjī", "भोपळा": "bhopla", "टुनुक": "ṭunuk",
                "एकी": "ekī", "काणी": "kāṇī", "वच्चुक": "vaccuka",
                "धूव": "dhūva", "सिंहु": "siṃhu", "वागु": "vāgu",
                "कोल्लो": "kollo", "म्हात्र": "mātra", "न्हयि": "nhayi",
                "धैर्यवान": "dhairyavān", "चिक्केचि": "cikkeci",
                "दक्ष": "dakṣa", "यज्ञ": "yajña", "सती": "satī",
                "नंदी": "nandī", "वीरभद्र": "vīrabhadra", "हिमालय": "himālaya",
                "मात्तें": "māttēṃ", "बोकड्येगले": "bokaḍyegale",
                "काय्ळो": "kāyḷo", "गुब्ची": "gubcī", "गूडु": "gūḍu",
                "गूडांतु": "gūḍāntu", "काय्ळ्यागलें": "kāyḷyāgalēṃ"
            };

            if (overrides[cleanWord]) {
                c.ai_iast = overrides[cleanWord];
            } else {
                c.ai_iast = iast;
            }
        }

        // Generate Usage IAST context for ALL entries (even UPDATEs if we have a context)
        if (c.usage_context) {
            c.ai_usage_iast = toIAST(c.usage_context);
        }
    });

    fs.writeFileSync(CANDIDATES_FILE, JSON.stringify(candidates, null, 2));
    console.log(`✅ Candidates augmented. Total checked: ${candidates.length}`);
}

augmentCandidates();
