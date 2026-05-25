import sys
import json
from playwright.sync_api import sync_playwright

def run():
    with sync_playwright() as p:
        # Use playwright's own chromium instead of local chrome
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto('http://localhost:3001')
        
        # 1. Page Title
        title = page.title()
        
        # 4. Theme (Check background color)
        bg_color = page.evaluate("window.getComputedStyle(document.body).backgroundColor")
        
        # 5. Console errors
        errors = []
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == 'error' else None)
        
        # Take screenshot
        page.screenshot(path='screenshot.png')
        
        # Header + Upload Text check via page text
        page_text = page.inner_text("body")
        
        results = {
            "title": title,
            "theme_bg": bg_color,
            "console_errors": errors,
            "has_audiovec_header": "audiovec" in page_text.lower(),
            "has_upload_text": "drop a wav file here" in page_text.lower()
        }
        print(json.dumps(results))
        browser.close()

if __name__ == "__main__":
    run()
