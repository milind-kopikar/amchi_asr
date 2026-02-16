const axios = require('axios');

async function countEntries(url) {
    try {
        const response = await axios.get(url);
        console.log(`URL: ${url}`);
        console.log(`Count: ${response.data.length}`);
        if (response.data.length > 0) {
            console.log(`First entry:`, JSON.stringify(response.data[0], null, 2));
            console.log(`Last entry:`, JSON.stringify(response.data[response.data.length - 1], null, 2));
        }
    } catch (error) {
        console.error(`Error fetching ${url}:`, error.message);
    }
}

const urls = [
    'https://konkani-dictionary-production.up.railway.app/api/dictionary',
    'https://konkanicollector-production.up.railway.app/api/dictionary'
];

urls.forEach(countEntries);
