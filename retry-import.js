const fs = require('fs');
const { importSpreadsheetDataEnhanced } = require('./enhanced-import.js');

/**
 * Retry import for entries that failed due to schema constraints
 */
async function retryFailedEntries() {
  try {
    console.log('🔄 Retrying import for previously failed entries...');
    
    // Check if error log exists
    const errorLogPath = 'import-errors.json';
    if (!fs.existsSync(errorLogPath)) {
      console.log('No error log found. Running fresh import...');
      return await importSpreadsheetDataEnhanced('./konkani_dictionary_csv.csv');
    }
    
    // Read previous errors
    const errors = JSON.parse(fs.readFileSync(errorLogPath, 'utf8'));
    console.log(`Found ${errors.length} previous errors`);
    
    // Filter for NULL constraint errors (these should work now)
    const nullErrors = errors.filter(err => 
      err.error.includes('null value in column "english_meaning"') ||
      err.error.includes('violates not-null constraint')
    );
    
    console.log(`${nullErrors.length} errors were due to NULL constraints (should work now)`);
    
    // Run fresh import to get all entries
    console.log('Running complete fresh import with fixed schema...');
    const result = await importSpreadsheetDataEnhanced('./konkani_dictionary_csv.csv');
    
    return result;
    
  } catch (error) {
    console.error('❌ Retry failed:', error);
    throw error;
  }
}

retryFailedEntries()
  .then(result => {
    console.log('✨ Retry completed!');
    console.log(`Final result: ${result.imported} imported, ${result.errors} errors`);
    process.exit(0);
  })
  .catch(error => {
    console.error('Failed:', error);
    process.exit(1);
  });