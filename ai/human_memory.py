# ai/human_memory.py - Human-like memory layer that integrates with existing MEGA-INTELLIGENT system
import json
import os
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import re
from ai.memory import get_user_memory, add_to_conversation_history

class HumanLikeMemory:
    """🧠 Human-like memory that integrates with your existing MEGA-INTELLIGENT system"""
    
    def __init__(self, username: str):
        self.username = username
        self.memory_dir = f"memory/{username}"
        os.makedirs(self.memory_dir, exist_ok=True)
        
        # Get the existing MEGA-INTELLIGENT memory system
        self.mega_memory = get_user_memory(username)
        
        # 3 Human-like Memory Systems (complement existing system)
        self.appointments = self.load_memory('human_appointments.json')
        self.life_events = self.load_memory('human_life_events.json') 
        self.conversation_highlights = self.load_memory('human_highlights.json')
        
        # Session tracking to avoid spam
        self.context_used_this_session = set()
        
        print(f"[HumanMemory] 🧠 Human-like memory layer initialized for {username}")
    
    def load_memory(self, filename: str) -> List[Dict]:
        """Load memory file"""
        file_path = os.path.join(self.memory_dir, filename)
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            print(f"[HumanMemory] ⚠️ Error loading {filename}: {e}")
            return []
    
    def save_memory(self, data: List[Dict], filename: str):
        """Save memory file"""
        file_path = os.path.join(self.memory_dir, filename)
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[HumanMemory] ❌ Error saving {filename}: {e}")
    
    def extract_and_store_human_memories(self, text: str):
        """🎯 Extract human-like memories ALONGSIDE existing MEGA-INTELLIGENT extraction"""
        
        # ALSO use the existing MEGA-INTELLIGENT extraction (don't replace it)
        self.mega_memory.extract_memories_from_text(text)
        
        text_lower = text.lower().strip()
        
        # Add our human-like patterns ON TOP of existing system
        self._extract_appointments(text, text_lower)
        self._extract_life_events(text, text_lower)
        self._extract_conversation_highlights(text, text_lower)
        
        # Save human-like memories
        self.save_memory(self.appointments, 'human_appointments.json')
        self.save_memory(self.life_events, 'human_life_events.json')
        self.save_memory(self.conversation_highlights, 'human_highlights.json')
    
    def _extract_appointments(self, text: str, text_lower: str):
        """📅 Extract appointments - focused on natural language"""
        appointment_patterns = [
            # With specific times
            (r'(?:dentist|doctor|appointment|meeting|interview|work) (?:tomorrow|today) at (\d{1,2}(?::\d{2})?(?:\s?(?:am|pm|AM|PM))?)', 'appointment'),
            (r'remind me (?:about\s+)?(.+?) (?:tomorrow|today) at (\d{1,2}(?::\d{2})?(?:\s?(?:am|pm|AM|PM))?)', 'reminder'),
            (r'i have (?:a\s+)?(.+?) (?:tomorrow|today) at (\d{1,2}(?::\d{2})?(?:\s?(?:am|pm|AM|PM))?)', 'appointment'),
            
            # Without specific times
            (r'(?:dentist|doctor|appointment|meeting|interview|work) (?:tomorrow|today)', 'appointment'),
            (r'i have (?:a\s+)?(.+?) (?:tomorrow|today)', 'appointment'),
        ]
        
        for pattern, event_type in appointment_patterns:
            match = re.search(pattern, text_lower)
            if match:
                if 'tomorrow' in text_lower:
                    event_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
                else:
                    event_date = datetime.now().strftime('%Y-%m-%d')
                
                # Extract topic and time intelligently
                if len(match.groups()) >= 2:
                    topic = match.group(1).strip() if match.group(1) else self._extract_topic_from_text(text)
                    time_str = match.group(2) if len(match.groups()) > 1 else None
                elif len(match.groups()) == 1:
                    # Could be topic or time
                    group_text = match.group(1)
                    if any(time_indicator in group_text for time_indicator in [':', 'am', 'pm', 'AM', 'PM']):
                        topic = self._extract_topic_from_text(text)
                        time_str = group_text
                    else:
                        topic = group_text.strip()
                        time_str = None
                else:
                    topic = self._extract_topic_from_text(text)
                    time_str = None
                
                appointment = {
                    'topic': topic,
                    'date': event_date,
                    'time': time_str,
                    'emotion': 'casual',
                    'status': 'pending',
                    'type': event_type,
                    'created': datetime.now().isoformat(),
                    'original_text': text
                }
                
                self.appointments.append(appointment)
                print(f"[HumanMemory] 📅 Stored appointment: {topic} on {event_date}" + (f" at {time_str}" if time_str else ""))
                break
    
    def _extract_life_events(self, text: str, text_lower: str):
        """📝 Extract life events with emotional intelligence"""
        life_event_patterns = [
            # Sensitive events
            (r'(?:i have|there\'s|going to) (?:a\s+)?funeral (?:tomorrow|today|this week)', 'funeral', 'sensitive'),
            (r'(?:my|our) (.+?) (?:died|passed away|passed)', 'death', 'sensitive'),
            (r'(?:i have|there\'s|going to) (?:a\s+)?wedding (?:tomorrow|today|this week)', 'wedding', 'happy'),
            
            # Medical events
            (r'(?:i have|i\'m having|going for) (?:surgery|operation|procedure) (?:tomorrow|today)', 'surgery', 'stressful'),
            (r'(?:going to|visiting|have to go to) (?:the\s+)?(?:hospital|doctor) (?:tomorrow|today)', 'medical_visit', 'stressful'),
            
            # Job/career events
            (r'(?:i have|there\'s|going to) (?:a\s+)?(?:job\s+)?interview (?:tomorrow|today)', 'job_interview', 'stressful'),
            (r'(?:starting|start) (?:new\s+)?job (?:tomorrow|today)', 'new_job', 'exciting'),
            
            # Family events
            (r'(?:my|our) (.+?) (?:is visiting|visiting|coming over) (?:tomorrow|today)', 'family_visit', 'happy'),
            
            # Ongoing emotional states
            (r'i\'m (?:really\s+)?stressed (?:about|out)', 'stress', 'supportive'),
            (r'(?:having|going through) (?:a\s+)?(?:hard|tough|difficult) time', 'difficult_period', 'supportive'),
        ]
        
        for pattern, topic, emotion in life_event_patterns:
            match = re.search(pattern, text_lower)
            if match:
                # Determine date
                if any(time_word in text_lower for time_word in ['tomorrow', 'today', 'this week']):
                    if 'tomorrow' in text_lower:
                        event_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
                    elif 'today' in text_lower:
                        event_date = datetime.now().strftime('%Y-%m-%d')
                    elif 'this week' in text_lower:
                        event_date = (datetime.now() + timedelta(days=3)).strftime('%Y-%m-%d')
                else:
                    event_date = datetime.now().strftime('%Y-%m-%d')
                
                # Extract specific details
                if match.groups() and match.group(1):
                    topic = f"{topic}_{match.group(1).strip()}"
                
                life_event = {
                    'topic': topic,
                    'date': event_date,
                    'emotion': emotion,
                    'status': 'pending',
                    'type': 'life_event',
                    'created': datetime.now().isoformat(),
                    'original_text': text
                }
                
                self.life_events.append(life_event)
                print(f"[HumanMemory] 📝 Stored life event: {topic} on {event_date} ({emotion})")
                break
    
    def _extract_conversation_highlights(self, text: str, text_lower: str):
        """💬 Extract small conversational highlights"""
        highlight_patterns = [
            (r'i\'m (?:really\s+)?(?:tired|exhausted|drained)', 'feeling_tired', 'supportive'),
            (r'i\'m (?:really\s+)?(?:excited|pumped|stoked) about (.+)', 'excited_about', 'happy'),
            (r'i (?:really\s+)?(?:love|like|enjoy) (.+)', 'likes', 'casual'),
            (r'i (?:really\s+)?(?:hate|dislike|can\'t stand) (.+)', 'dislikes', 'casual'),
            (r'i\'m thinking about (.+)', 'considering', 'casual'),
            (r'i want to (?:change|quit|leave) (?:my\s+)?job', 'job_change_thoughts', 'supportive'),
            (r'i\'m (?:worried|concerned|anxious) about (.+)', 'worried_about', 'supportive'),
            (r'looking forward to (.+)', 'anticipating', 'happy'),
        ]
        
        for pattern, topic, emotion in highlight_patterns:
            match = re.search(pattern, text_lower)
            if match:
                highlight = {
                    'topic': topic,
                    'detail': match.group(1) if match.groups() else text,
                    'emotion': emotion,
                    'status': 'noted',
                    'created': datetime.now().isoformat(),
                    'original_text': text
                }
                
                self.conversation_highlights.append(highlight)
                print(f"[HumanMemory] 💬 Noted highlight: {topic}")
                break
    
    def _extract_topic_from_text(self, text: str) -> str:
        """Extract topic from text when not captured by pattern"""
        text_lower = text.lower()
        
        if 'dentist' in text_lower:
            return 'dentist'
        elif 'doctor' in text_lower:
            return 'doctor'
        elif 'meeting' in text_lower:
            return 'meeting'
        elif 'interview' in text_lower:
            return 'interview'
        elif 'work' in text_lower:
            return 'work'
        else:
            return 'appointment'
    
    def check_for_natural_context_response(self) -> Optional[str]:
        """🎯 Check if we should naturally bring up memories"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Check appointments first (time-sensitive)
        response = self._check_appointments(today)
        if response:
            return response
        
        # Check life events 
        response = self._check_life_events(today)
        if response:
            return response
        
        # Check conversation highlights (occasionally)
        if random.random() < 0.2:  # 20% chance
            response = self._check_conversation_highlights()
            if response:
                return response
        
        return None
    
    def _check_appointments(self, today: str) -> Optional[str]:
        """📅 Check appointments with natural responses"""
        for appointment in self.appointments:
            if appointment['date'] == today and appointment['status'] == 'pending':
                topic = appointment['topic']
                time_str = appointment.get('time', '')
                
                # Avoid repeating in same session
                context_key = f"appointment_{appointment['date']}_{topic}"
                if context_key in self.context_used_this_session:
                    continue
                
                self.context_used_this_session.add(context_key)
                
                # Natural, caring responses
                if time_str:
                    if 'dentist' in topic.lower():
                        responses = [
                            f"Yo! Just a heads up — dentist at {time_str}. You ready for the drill? 😆",
                            f"Hey, don't forget your dentist appointment at {time_str}! Try not to chicken out 😂",
                            f"Quick reminder: dentist at {time_str} today. Hope they go easy on ya!",
                        ]
                    else:
                        responses = [
                            f"Heads up — {topic} at {time_str} today. You got this!",
                            f"Don't forget your {topic} at {time_str}!",
                            f"Just a reminder: {topic} at {time_str} today 💪",
                        ]
                else:
                    responses = [
                        f"Hey, you've got {topic} today. How you feeling about it?",
                        f"Don't forget about {topic} today!",
                        f"Hope your {topic} goes well today!"
                    ]
                
                appointment['status'] = 'reminded'
                self.save_memory(self.appointments, 'human_appointments.json')
                
                return random.choice(responses)
        
        return None
    
    def _check_life_events(self, today: str) -> Optional[str]:
        """📝 Check life events with emotional intelligence"""
        for event in self.life_events:
            if event['date'] == today and event['status'] == 'pending':
                topic = event['topic']
                emotion = event['emotion']
                
                context_key = f"life_event_{event['date']}_{topic}"
                if context_key in self.context_used_this_session:
                    continue
                
                self.context_used_this_session.add(context_key)
                
                if emotion == 'sensitive':
                    responses = [
                        f"Hey, how'd the {topic} go? You alright? ❤️",
                        f"Just checking — how was the {topic}? You doing okay?",
                        f"How are you holding up after the {topic}?"
                    ]
                elif emotion == 'stressful':
                    responses = [
                        f"So how'd the {topic} go? You survive? 😅",
                        f"How was the {topic}? Hope it went better than expected!",
                        f"Did the {topic} go okay? How you feeling?"
                    ]
                elif emotion == 'happy' or emotion == 'exciting':
                    responses = [
                        f"Yooo! How was the {topic}? Tell me everything! 😄",
                        f"How'd the {topic} go?! Was it awesome?",
                        f"So how was the {topic}? I wanna hear all about it!"
                    ]
                else:  # casual
                    responses = [
                        f"How'd the {topic} go?",
                        f"So, how was your {topic}?",
                        f"How was the {topic} today?"
                    ]
                
                event['status'] = 'followed_up'
                self.save_memory(self.life_events, 'human_life_events.json')
                
                return random.choice(responses)
        
        return None
    
    def _check_conversation_highlights(self) -> Optional[str]:
        """💬 Occasionally bring up conversation highlights"""
        recent_highlights = []
        cutoff_date = datetime.now() - timedelta(days=3)
        
        for highlight in self.conversation_highlights:
            created = datetime.fromisoformat(highlight['created'])
            if created >= cutoff_date and highlight['status'] == 'noted':
                context_key = f"highlight_{highlight['topic']}_{highlight['created'][:10]}"
                if context_key not in self.context_used_this_session:
                    recent_highlights.append(highlight)
        
        if recent_highlights:
            highlight = random.choice(recent_highlights)
            topic = highlight['topic']
            detail = highlight.get('detail', '')
            
            context_key = f"highlight_{topic}_{highlight['created'][:10]}"
            self.context_used_this_session.add(context_key)
            
            if topic == 'feeling_tired':
                responses = [
                    "Hey, you feeling less tired today?",
                    "How's your energy been lately?",
                ]
            elif topic == 'job_change_thoughts':
                responses = [
                    "Still thinking about that job change?",
                    "How are you feeling about the job situation?",
                ]
            elif topic == 'excited_about':
                responses = [
                    f"How's that {detail} thing going you were excited about?",
                    f"Still pumped about {detail}?",
                ]
            else:
                responses = [
                    f"Hey, how's that {detail} situation?",
                    f"Any updates on {detail}?"
                ]
            
            highlight['status'] = 'followed_up'
            self.save_memory(self.conversation_highlights, 'human_highlights.json')
            
            return random.choice(responses)
        
        return None
    
    def reset_session_context(self):
        """Reset session context (call when new conversation starts)"""
        self.context_used_this_session.clear()
        print(f"[HumanMemory] 🔄 Session context reset for {self.username}")