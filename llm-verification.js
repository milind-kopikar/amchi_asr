const { TEST_CASES, WATER_TEST_CASES, LLM_TEST_QUERIES } = require("../../test-cases");

// Function to verify LLM responses against real database entries
function verifyLLMResponse(llmResponse, expectedEntryNumbers) {
  const results = {
    passed: true,
    issues: [],
    validEntries: [],
    invalidEntries: []
  };

  if (!llmResponse.results || llmResponse.results.length === 0) {
    results.passed = false;
    results.issues.push("No results returned from LLM");
    return results;
  }

  llmResponse.results.forEach(entry => {
    const isValid = expectedEntryNumbers.includes(entry.entry_number);
    
    if (isValid) {
      results.validEntries.push(entry.entry_number);
    } else {
      results.invalidEntries.push({
        entry_number: entry.entry_number,
        word: entry.word_konkani_devanagari,
        meaning: entry.english_meaning
      });
      results.passed = false;
      results.issues.push(`Unexpected entry ${entry.entry_number}: ${entry.word_konkani_devanagari}`);
    }
  });

  return results;
}

module.exports = {
  verifyLLMResponse
};
