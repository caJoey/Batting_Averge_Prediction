/*
Put together basic stats  from baseball-reference 
 - Scrape raw csv from baseball-reference then send to process_csv.py for 
preprocessing and appending to stats.csv
*/

import { chromium } from 'playwright';
import { spawn } from "child_process";

// removes first and last lines
function processStats(stats_string) {
    const stats_arr = stats_string.split('\n');
    // first and last lines arent important
    return stats_arr.slice(1, -1).join('\n');
}

// spawns python child to write to csv
async function spawn_python(csv_text, year) {
    const child = spawn('python', ['process_csv.py', year]);
    // send in the text
    child.stdin.write(csv_text);
    // adds EOF to child's stdin, making it so read command doesn't hang
    child.stdin.end();
    child.stderr.on('data', (data) => {
        console.error(`[Python error][${year}] ${data.toString()}`);
    });
    // wait for the child file to end before returning (csv will be appended)
    await new Promise((resolve) => {
        child.on('close', (code) => {
            if (code !== 0) {
                console.error(`[Python exited with code ${code}] for year ${year}`);
            }
            resolve();
        });
    });
}

async function getStats() {
    const browser = await chromium.launch({ headless: true });
    const page = await browser.newPage();
    await page.setExtraHTTPHeaders({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9'
    });
    const START_YEAR = 1955;
    const END_YEAR = 2025;
    for (let year = START_YEAR; year <= END_YEAR; year++) {
        await page.goto(`https://www.baseball-reference.com/leagues/majors/${year}-standard-batting.shtml`, { waitUntil: 'domcontentloaded' });
        // wait until table is loaded so function works
        await page.waitForSelector('#share_on_players_standard_batting');
        const stats_string = await page.evaluate(() => {
            return (get_csv_output('players_standard_batting', false, true, false));
        });
        const csv_text = processStats(stats_string);
        await spawn_python(csv_text, year);
    }
    await browser.close();
}

getStats();
