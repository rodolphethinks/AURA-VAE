import requests
import time

BOT_TOKEN = "8225968498:AAFiZUsJbIdpENP73vh_rs0k-j8aLt0x3nQ"
URL = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"

print(f"Checking for messages to Bot {BOT_TOKEN[:10]}...")
print("1. Please send a message (e.g., 'Hello') to your bot @AuraFleetBot on Telegram.")
print("2. Or add the bot to a Group/Channel and send a message there.")
print("Waiting...")

try:
    response = requests.get(URL)
    data = response.json()
    
    if not data['ok']:
        print(f"Error: {data}")
        exit()
        
    results = data['result']
    
    if not results:
        print("\nNo messages found yet.")
        print("-> Go to Telegram, find @AuraFleetBot, click START or send a message.")
        print("-> Then run this script again.")
    else:
        print("\n" + "="*40)
        print("FOUND EXTANT CHATS:")
        print("="*40)
        
        last_chat = None
        for update in results:
            if 'message' in update:
                chat = update['message']['chat']
                chat_id = chat['id']
                chat_type = chat['type']
                chat_title = chat.get('title', chat.get('username', 'Private'))
                
                print(f"Chat ID: {chat_id}")
                print(f"Type:    {chat_type}")
                print(f"Name:    {chat_title}")
                print("-" * 20)
                last_chat = chat_id
            
            elif 'channel_post' in update:
                chat = update['channel_post']['chat']
                chat_id = chat['id']
                chat_type = chat['type']
                chat_title = chat.get('title', 'Channel')
                
                print(f"Chat ID: {chat_id}")
                print(f"Type:    {chat_type}")
                print(f"Name:    {chat_title}")
                print("-" * 20)
                last_chat = chat_id
                
        if last_chat:
            print(f"\nRecommended Action: Copy '{last_chat}' into CHAT_ID in DataCollectionService.java")

except Exception as e:
    print(f"Error: {e}")
