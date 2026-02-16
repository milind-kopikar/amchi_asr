#!/usr/bin/env node

/**
 * Script to update API URL in GitHub Pages files after Railway deployment
 * Usage: node scripts/update-api-url.js https://your-railway-app.railway.app
 */

const fs = require('fs');
const path = require('path');

function updateApiUrl(newApiUrl) {
  const githubPagesPath = path.join(__dirname, '..', 'github-pages', 'index.html');
  
  try {
    console.log('🔄 Updating API URL in GitHub Pages files...');
    
    // Read the file
    let content = fs.readFileSync(githubPagesPath, 'utf8');
    
    // Replace the API_BASE URL
    const oldApiPattern = /const API_BASE = ['"`]https:\/\/.*?['"`];/;
    const newApiLine = `const API_BASE = '${newApiUrl}/api';`;
    
    if (oldApiPattern.test(content)) {
      content = content.replace(oldApiPattern, newApiLine);
      console.log('✅ Found and updated existing API URL');
    } else {
      // If pattern not found, look for the placeholder
      const placeholderPattern = /const API_BASE = ['"`]https:\/\/your-railway-app\.railway\.app\/api['"`];/;
      if (placeholderPattern.test(content)) {
        content = content.replace(placeholderPattern, newApiLine);
        console.log('✅ Updated placeholder API URL');
      } else {
        console.log('⚠️ Could not find API_BASE declaration to update');
        console.log('Please manually update the API_BASE constant in github-pages/index.html');
        return;
      }
    }
    
    // Write back to file
    fs.writeFileSync(githubPagesPath, content);
    
    console.log(`✅ Updated API URL to: ${newApiUrl}/api`);
    console.log('📁 File updated: github-pages/index.html');
    console.log('');
    console.log('Next steps:');
    console.log('1. Copy github-pages/index.html to your moksh-kopikar.github.io/konkani-dictionary/ folder');
    console.log('2. Copy github-pages/homepage-card.html content to your homepage');
    console.log('3. Commit and push to GitHub Pages');
    
  } catch (error) {
    console.error('❌ Error updating API URL:', error.message);
  }
}

// Get API URL from command line arguments
const newApiUrl = process.argv[2];

if (!newApiUrl) {
  console.log('📋 Usage: node scripts/update-api-url.js <railway-url>');
  console.log('📋 Example: node scripts/update-api-url.js https://konkani-dictionary-production.up.railway.app');
  process.exit(1);
}

// Validate URL format
try {
  new URL(newApiUrl);
} catch {
  console.error('❌ Invalid URL format. Please provide a valid Railway URL.');
  process.exit(1);
}

updateApiUrl(newApiUrl);