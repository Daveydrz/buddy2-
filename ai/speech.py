# ai/speech.py - Speech-to-text processing
import asyncio
import json
import numpy as np
import websockets
from config import FASTER_WHISPER_WS, DEBUG
import re
import os
from datetime import datetime

async def whisper_stt_async(audio):
    """Transcribe audio using Whisper WebSocket"""
    try:
        if audio.dtype != np.int16:
            if np.issubdtype(audio.dtype, np.floating):
                audio = (audio * 32767).clip(-32768, 32767).astype(np.int16)
            else:
                audio = audio.astype(np.int16)
        
        async with websockets.connect(FASTER_WHISPER_WS, ping_interval=None) as ws:
            await ws.send(audio.tobytes())
            await ws.send("end")
            
            try:
                message = await asyncio.wait_for(ws.recv(), timeout=15)
            except asyncio.TimeoutError:
                print("[Buddy V2] Whisper timeout")
                return ""
            
            try:
                data = json.loads(message)
                text = data.get("text", "").strip()
                if DEBUG:
                    print(f"[Buddy V2] 📝 Whisper: '{text}'")
                return text
            except:
                text = message.decode("utf-8") if isinstance(message, bytes) else message
                if DEBUG:
                    print(f"[Buddy V2] 📝 Whisper: '{text}'")
                return text.strip()
                
    except Exception as e:
        print(f"[Buddy V2] Whisper error: {e}")
        return ""

def extract_spoken_name(text: str, system_username: str) -> str:
    """Extract the user's actual spoken name, not system username"""
    
    # Look for "I'm [Name]" or "My name is [Name]"
    name_patterns = [
        r"i'm\s+([a-zA-Z]+)",
        r"my\s+name\s+is\s+([a-zA-Z]+)",
        r"call\s+me\s+([a-zA-Z]+)",
        r"i\s+am\s+([a-zA-Z]+)"
    ]
    
    for pattern in name_patterns:
        match = re.search(pattern, text.lower())
        if match:
            spoken_name = match.group(1).capitalize()
            
            # Don't use system usernames as spoken names
            if spoken_name.lower() != system_username.lower():
                return spoken_name
    
    return None

def set_primary_identity(system_username: str, spoken_name: str):
    """Map system username to spoken name"""
    
    identity_file = f"memory/{system_username}/primary_identity.json"
    os.makedirs(f"memory/{system_username}", exist_ok=True)
    
    identity_data = {
        "system_username": system_username,
        "spoken_name": spoken_name,
        "display_name": spoken_name,
        "created_date": datetime.now().isoformat()
    }
    
    with open(identity_file, 'w') as f:
        json.dump(identity_data, f, indent=2)
    
    print(f"[Identity] Set primary identity: {system_username} → {spoken_name}")

def get_display_name(system_username: str) -> str:
    """Get the user's preferred display name"""
    
    identity_file = f"memory/{system_username}/primary_identity.json"
    
    if os.path.exists(identity_file):
        with open(identity_file, 'r') as f:
            identity = json.load(f)
            return identity.get("spoken_name", system_username)
    
    return system_username

def identify_user(spoken_input: str, system_username: str) -> str:
    """Identify user and prevent duplicates"""
    
    # Check if user introduced themselves
    spoken_name = extract_spoken_name(spoken_input, system_username)
    
    if spoken_name:
        # Set or update their primary identity
        set_primary_identity(system_username, spoken_name)
        print(f"[Identity] User identified as: {spoken_name} (system: {system_username})")
    
    # Always return the system username for memory storage
    # But display the spoken name in responses
    return system_username

def transcribe_audio(audio):
    """Synchronous wrapper for Whisper STT"""
    return asyncio.run(whisper_stt_async(audio))