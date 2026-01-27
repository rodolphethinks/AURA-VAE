import requests
import os
import json
from datetime import datetime

BOT_TOKEN = "8225968498:AAFiZUsJbIdpENP73vh_rs0k-j8aLt0x3nQ"
BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SAVE_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

def get_latest_audio():
    # Get updates
    print(f"Checking for updates from bot...")
    response = requests.get(f"{BASE_URL}/getUpdates")
    data = response.json()
    
    if not data.get("ok"):
        print(f"Error getting updates: {data}")
        return

    results = data.get("result", [])
    if not results:
        print("No updates found.")
        return

    print(f"Found {len(results)} updates. Searching for audio...")

    # Write debug log
    with open("telegram_debug.json", "w") as f:
        json.dump(results, f, indent=2)

    # Iterate backwards to find latest audio
    for update in reversed(results):
        message = update.get("message") or update.get("channel_post")
        if not message:
            continue
            
        file_id = None
        file_name = None
        
        # Check for audio types
        keys = list(message.keys())
        print(f"Message keys: {keys}")
        
        if "audio" in message:
            print("Found Audio message")
            audio = message["audio"]
            file_id = audio["file_id"]
            file_name = audio.get("file_name", "downloaded_audio.mp3")
        elif "voice" in message:
            print("Found Voice message")
            voice = message["voice"]
            file_id = voice["file_id"]
            file_name = f"voice_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ogg"
        elif "document" in message:
            doc = message["document"]
            mime = doc.get("mime_type", "")
            if "audio" in mime or "application/octet-stream" in mime: # Octet stream is common for generic uploads
                 print(f"Found Document (mime: {mime})")
                 file_id = doc["file_id"]
                 file_name = doc.get("file_name", "downloaded_doc.audio")
        
        if file_id:
            download_file(file_id, file_name)
            return

    print("No audio found in recent updates.")

def download_file(file_id, file_name):
    # Get file path
    print(f"Getting file info for ID: {file_id}...")
    res = requests.get(f"{BASE_URL}/getFile?file_id={file_id}")
    file_info = res.json()
    
    if not file_info.get("ok"):
        print(f"Error getting file info: {file_info}")
        return
        
    file_path_remote = file_info["result"]["file_path"]
    download_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path_remote}"
    
    # Handle duplicates
    save_path = os.path.join(SAVE_DIR, file_name)
    base, ext = os.path.splitext(save_path)
    counter = 1
    while os.path.exists(save_path):
        save_path = f"{base}_{counter}{ext}"
        counter += 1

    print(f"Downloading to {save_path}...")
    try:
        file_content = requests.get(download_url).content
        
        # Ensure dir exists
        os.makedirs(SAVE_DIR, exist_ok=True)
        
        with open(save_path, "wb") as f:
            f.write(file_content)
        
        print(f"Successfully saved: {os.path.basename(save_path)}")
        
        # Print file size
        size_kb = os.path.getsize(save_path) / 1024
        print(f"Size: {size_kb:.2f} KB")
        
    except Exception as e:
        print(f"Failed to download: {e}")

if __name__ == "__main__":
    get_latest_audio()
