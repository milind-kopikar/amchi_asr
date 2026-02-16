// Test cases extracted from Railway database
// These are REAL entries that the LLM should return (no hallucination)

const TEST_CASES = {
  // Test Case 1: Entry #2
  "aamchigele": {
    entry_number: 2,
    word_konkani_devanagari: "आम्चिगेले",
    word_konkani_english_alphabet: "aamchigele", 
    english_meaning: "fellow konkanis",
    context_usage_sentence: "आम्चिगेले उल्लव्य"
  },

  // Test Case 2: Entry #3  
  "amchikele": {
    entry_number: 3,
    word_konkani_devanagari: "अम्चिकेले",
    word_konkani_english_alphabet: "[अम्चिकेले]",
    english_meaning: "fellow konkanis", 
    context_usage_sentence: null
  },

  // Test Case 3: Entry #4
  "kalosk": {
    entry_number: 4,
    word_konkani_devanagari: "काळोस्क",
    word_konkani_english_alphabet: "[काळोस्क]",
    english_meaning: "dark",
    context_usage_sentence: null
  },

  // Test Case 4: Entry #5
  "angadi": {
    entry_number: 5,
    word_konkani_devanagari: "आंगडी", 
    word_konkani_english_alphabet: "[आंगडी]",
    english_meaning: "shop",
    context_usage_sentence: null
  },

  // Test Case 5: Entry #6
  "angan": {
    entry_number: 6,
    word_konkani_devanagari: "आंगणं",
    word_konkani_english_alphabet: "[आंगणं]", 
    english_meaning: "courtyard",
    context_usage_sentence: null
  }
};

// Water-related test cases (from previous search)
const WATER_TEST_CASES = {
  "uddaak": {
    entry_number: 2336,
    word_konkani_devanagari: "उद्दाक",
    word_konkani_english_alphabet: "[उद्दाक]",
    english_meaning: "water",
    context_usage_sentence: null
  },
  
  "uddha": {
    entry_number: 2339,
    word_konkani_devanagari: "उद्द्हा", 
    word_konkani_english_alphabet: "[उद्द्हा]",
    english_meaning: "water",
    context_usage_sentence: null
  }
};

// Test queries for LLM validation
const LLM_TEST_QUERIES = [
  {
    query: "What is shop in Konkani?",
    expected_entries: [5], // Should return आंगडी
    should_not_contain: ["made up words", "hallucinated entries"]
  },
  
  {
    query: "What is water in Konkani?", 
    expected_entries: [2336, 2339], // Should return उद्दाक, उद्द्हा
    should_not_contain: ["उद्दाक", "जळ", "नीर"] // These don't exist in DB? (keep for test)
  },
  
  {
    query: "Konkani word for courtyard",
    expected_entries: [6], // Should return आंगणं
    should_not_contain: ["hallucinated words"]
  }
];

module.exports = {
  TEST_CASES,
  WATER_TEST_CASES, 
  LLM_TEST_QUERIES
};
