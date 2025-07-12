# ai/memory.py - MEGA-INTELLIGENT Memory System with Advanced Context Awareness
import time
import json
import datetime
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from config import MAX_HISTORY_LENGTH, DEBUG
from enum import Enum

# Enhanced settings with fallbacks
try:
    from config import (ENHANCED_CONVERSATION_MEMORY, CONVERSATION_MEMORY_LENGTH, 
                       CONVERSATION_CONTEXT_LENGTH, CONVERSATION_SUMMARY_ENABLED,
                       CONVERSATION_SUMMARY_THRESHOLD, TOPIC_TRACKING_ENABLED,
                       MAX_CONVERSATION_TOPICS, CONTEXT_COMPRESSION_ENABLED,
                       MAX_CONTEXT_TOKENS)
except ImportError:
    ENHANCED_CONVERSATION_MEMORY = True
    CONVERSATION_MEMORY_LENGTH = 25
    CONVERSATION_CONTEXT_LENGTH = 10
    CONVERSATION_SUMMARY_ENABLED = True
    CONVERSATION_SUMMARY_THRESHOLD = 18
    TOPIC_TRACKING_ENABLED = True
    MAX_CONVERSATION_TOPICS = 6
    CONTEXT_COMPRESSION_ENABLED = True
    MAX_CONTEXT_TOKENS = 1500

# 🧠 MEGA-INTELLIGENT MEMORY ENHANCEMENTS
class EntityStatus(Enum):
    """Track current status of entities"""
    CURRENT = "current"
    FORMER = "former"
    DECEASED = "deceased"
    SOLD = "sold"
    LOST = "lost"
    ENDED = "ended"
    UNKNOWN = "unknown"

class EmotionalImpact(Enum):
    """Emotional significance levels"""
    CRITICAL = 0.9  # Death, major life events
    HIGH = 0.7      # Job loss, breakups
    MEDIUM = 0.5    # Moving, minor changes
    LOW = 0.3       # Preferences, activities
    MINIMAL = 0.1   # Basic facts

@dataclass
class EntityMemory:
    """🧠 MEGA-INTELLIGENT: Track entities with full context"""
    name: str
    entity_type: str  # "pet", "person", "possession", "relationship"
    status: EntityStatus
    emotional_significance: float
    date_learned: str
    last_updated: str
    context_description: str
    related_memories: List[str]
    emotional_context: List[str]  # ["beloved", "family", "missed"]
    temporal_context: Optional[str] = None  # "last week", "yesterday"
    
@dataclass
class LifeEvent:
    """🧠 MEGA-INTELLIGENT: Track major life events"""
    event_type: str  # "death", "birth", "job_change", "relationship_change"
    description: str
    entities_involved: List[str]
    emotional_impact: float
    date_occurred: str
    date_learned: str
    ongoing_effects: List[str]  # ["grieving", "adjusting", "celebrating"]
    follow_up_contexts: List[str]  # ["avoid_suggesting_activities", "offer_support"]

@dataclass
class PersonalFact:
    """Enhanced personal fact storage"""
    category: str
    key: str
    value: str
    confidence: float
    date_learned: str
    last_mentioned: str
    source_context: str
    emotional_weight: float = 0.3
    current_status: EntityStatus = EntityStatus.CURRENT
    related_entities: List[str] = None
    
    def __post_init__(self):
        if self.related_entities is None:
            self.related_entities = []

@dataclass
class EmotionalState:
    """Enhanced emotional state tracking"""
    emotion: str
    intensity: int
    context: str
    date: str
    follow_up_needed: bool
    related_memories: List[str] = None
    trigger_entities: List[str] = None
    
    def __post_init__(self):
        if self.related_memories is None:
            self.related_memories = []
        if self.trigger_entities is None:
            self.trigger_entities = []

@dataclass
class ScheduledEvent:
    """Enhanced event tracking"""
    event_type: str
    description: str
    date: str
    reminder_dates: List[str]
    completed: bool
    emotional_significance: float = 0.5
    related_entities: List[str] = None
    
    def __post_init__(self):
        if self.related_entities is None:
            self.related_entities = []

@dataclass
class ConversationTopic:
    """Enhanced topic tracking"""
    topic: str
    start_time: str
    last_mentioned: str
    message_count: int
    keywords: List[str]
    emotional_context: str = "neutral"
    related_entities: List[str] = None
    
    def __post_init__(self):
        if self.related_entities is None:
            self.related_entities = []

class MemoryContextValidator:
    """🧠 MEGA-INTELLIGENT: Validate memory context and prevent inappropriate responses"""
    
    def __init__(self):
        self.inappropriate_suggestions = {
            EntityStatus.DECEASED: [
                "walk", "play", "feed", "pet", "visit", "call", "talk to", 
                "bring", "take out", "exercise", "groom", "train"
            ],
            EntityStatus.SOLD: [
                "drive", "use", "fix", "maintain", "clean"
            ],
            EntityStatus.ENDED: [
                "contact", "call", "visit", "meet", "date", "spend time"
            ],
            EntityStatus.LOST: [
                "use", "find", "get", "bring"
            ]
        }
    
    def validate_response_appropriateness(self, proposed_response: str, entities: Dict[str, EntityMemory]) -> Tuple[bool, List[str]]:
        """Check if proposed response conflicts with known entity statuses"""
        warnings = []
        response_lower = proposed_response.lower()
        
        for entity_name, entity in entities.items():
            if entity_name.lower() in response_lower:
                if entity.status in self.inappropriate_suggestions:
                    inappropriate_words = self.inappropriate_suggestions[entity.status]
                    for word in inappropriate_words:
                        if word in response_lower:
                            warnings.append(f"Inappropriate suggestion '{word}' for {entity.status.value} entity '{entity_name}'")
        
        return len(warnings) == 0, warnings
    
    def suggest_appropriate_response_context(self, entity: EntityMemory) -> List[str]:
        """Suggest appropriate response contexts for entities"""
        suggestions = []
        
        if entity.status == EntityStatus.DECEASED:
            if entity.emotional_significance > 0.7:
                suggestions.extend([
                    "offer_condolences", "share_memory", "acknowledge_loss", 
                    "express_empathy", "avoid_present_tense"
                ])
        elif entity.status == EntityStatus.FORMER:
            suggestions.extend([
                "use_past_tense", "acknowledge_change", "offer_support"
            ])
        elif entity.status == EntityStatus.ENDED:
            suggestions.extend([
                "acknowledge_end", "offer_emotional_support", "avoid_couple_activities"
            ])
        
        return suggestions

class MemoryInferenceEngine:
    """🧠 MEGA-INTELLIGENT: Make logical inferences from stored memories"""
    
    def infer_entity_implications(self, entity: EntityMemory) -> List[str]:
        """Infer logical implications of entity status"""
        implications = []
        
        if entity.status == EntityStatus.DECEASED:
            implications.extend([
                "no_physical_activities",
                "use_past_tense",
                "emotional_sensitivity_required",
                "grief_support_appropriate"
            ])
        elif entity.status == EntityStatus.FORMER:
            implications.extend([
                "relationship_ended",
                "use_past_tense",
                "avoid_current_references"
            ])
        elif entity.status == EntityStatus.SOLD:
            implications.extend([
                "no_longer_owned",
                "past_ownership_only"
            ])
        
        return implications
    
    def detect_memory_contradictions(self, new_fact: PersonalFact, existing_entities: Dict[str, EntityMemory]) -> List[str]:
        """Detect contradictions between new information and existing memories"""
        contradictions = []
        
        # Check if new fact contradicts existing entity status
        for entity_name, entity in existing_entities.items():
            if entity_name.lower() in new_fact.value.lower():
                if entity.status == EntityStatus.DECEASED and "alive" in new_fact.value.lower():
                    contradictions.append(f"New fact suggests {entity_name} is alive, but recorded as deceased")
                elif entity.status == EntityStatus.CURRENT and "used to" in new_fact.key:
                    contradictions.append(f"New fact suggests {entity_name} is former, but recorded as current")
        
        return contradictions

class UserMemorySystem:
    """🧠 MEGA-INTELLIGENT: Enhanced memory system with context awareness"""
    
    def __init__(self, username: str):
        self.username = username
        self.memory_dir = Path(f"memory/{username}")
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # Enhanced memory storage
        self.personal_facts: Dict[str, PersonalFact] = {}
        self.emotional_history: List[EmotionalState] = []
        self.scheduled_events: List[ScheduledEvent] = []
        self.conversation_topics: List[ConversationTopic] = []
        
        # 🧠 NEW: Advanced memory components
        self.entity_memories: Dict[str, EntityMemory] = {}
        self.life_events: List[LifeEvent] = {}
        self.memory_validator = MemoryContextValidator()
        self.inference_engine = MemoryInferenceEngine()
        
        self.load_memory()
        print(f"[MegaMemory] 🧠 MEGA-INTELLIGENT memory system loaded for {username}")
    
    def add_entity_memory(self, name: str, entity_type: str, status: EntityStatus, 
                         emotional_significance: float, context: str):
        """🧠 Add or update entity memory with full context"""
        current_time = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        
        entity = EntityMemory(
            name=name,
            entity_type=entity_type,
            status=status,
            emotional_significance=emotional_significance,
            date_learned=current_time,
            last_updated=current_time,
            context_description=context,
            related_memories=[],
            emotional_context=self._extract_emotional_context(context)
        )
        
        self.entity_memories[name.lower()] = entity
        
        # 🧠 MEGA-INTELLIGENT: Auto-generate implications
        implications = self.inference_engine.infer_entity_implications(entity)
        
        print(f"[MegaMemory] 🧠 Entity Added: {name} ({entity_type}) - Status: {status.value}")
        print(f"[MegaMemory] 💭 Implications: {', '.join(implications)}")
        
        self.save_memory()
    
    def add_life_event(self, event_type: str, description: str, entities_involved: List[str], 
                      emotional_impact: float, ongoing_effects: List[str] = None):
        """🧠 Record major life events with context"""
        current_time = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        
        if ongoing_effects is None:
            ongoing_effects = []
        
        event = LifeEvent(
            event_type=event_type,
            description=description,
            entities_involved=entities_involved,
            emotional_impact=emotional_impact,
            date_occurred=current_time,
            date_learned=current_time,
            ongoing_effects=ongoing_effects,
            follow_up_contexts=self._generate_follow_up_contexts(event_type, emotional_impact)
        )
        
        event_id = f"{event_type}_{int(time.time())}"
        self.life_events[event_id] = event
        
        # Update related entities
        for entity_name in entities_involved:
            if entity_name.lower() in self.entity_memories:
                self.entity_memories[entity_name.lower()].related_memories.append(event_id)
        
        print(f"[MegaMemory] 📅 Life Event: {description} (Impact: {emotional_impact})")
        self.save_memory()
    
    def validate_response_before_output(self, proposed_response: str) -> Tuple[bool, str]:
        """🧠 MEGA-INTELLIGENT: Validate response appropriateness before output"""
        is_appropriate, warnings = self.memory_validator.validate_response_appropriateness(
            proposed_response, self.entity_memories
        )
        
        if not is_appropriate:
            print(f"[MegaMemory] ⚠️ Response validation failed: {warnings}")
            
            # Generate alternative response
            alternative = self._generate_appropriate_alternative(proposed_response, warnings)
            return False, alternative
        
        return True, proposed_response
    
    def _generate_appropriate_alternative(self, original_response: str, warnings: List[str]) -> str:
        """Generate contextually appropriate alternative response"""
        # This is a simplified version - in a full implementation, 
        # this would use more sophisticated NLP to rewrite responses
        
        for warning in warnings:
            if "deceased" in warning and "walk" in warning:
                return "I remember you mentioned your pet. Losing a beloved companion is really difficult. How are you feeling about that?"
            elif "deceased" in warning:
                return "I'm thinking of the loss you mentioned. It's completely normal to still feel emotional about it."
        
        return "I understand. Would you like to talk about how you're feeling today?"
    
    def _extract_emotional_context(self, context: str) -> List[str]:
        """Extract emotional context tags from text"""
        emotional_indicators = {
            "beloved": ["love", "adore", "cherish", "dear", "precious"],
            "missed": ["miss", "gone", "absence", "empty"],
            "sad": ["sad", "grief", "sorrow", "heartbreak"],
            "happy": ["joy", "wonderful", "amazing", "great"],
            "family": ["family", "relative", "close", "important"]
        }
        
        context_lower = context.lower()
        tags = []
        
        for tag, indicators in emotional_indicators.items():
            if any(indicator in context_lower for indicator in indicators):
                tags.append(tag)
        
        return tags
    
    def _generate_follow_up_contexts(self, event_type: str, emotional_impact: float) -> List[str]:
        """Generate appropriate follow-up contexts for events"""
        contexts = []
        
        if event_type == "death" and emotional_impact > 0.7:
            contexts.extend([
                "offer_emotional_support",
                "avoid_suggesting_activities_with_deceased",
                "acknowledge_grief_process",
                "be_sensitive_to_mentions"
            ])
        elif event_type == "relationship_end":
            contexts.extend([
                "avoid_couple_activities",
                "use_past_tense_for_ex",
                "offer_emotional_support"
            ])
        elif event_type == "job_loss":
            contexts.extend([
                "avoid_work_related_assumptions",
                "offer_career_support",
                "be_sensitive_to_stress"
            ])
        
        return contexts
    
    def get_contextual_memory_for_response(self) -> str:
        """🧠 Get memory context optimized for appropriate responses"""
        context_parts = []
        
        # Recent personal facts with entity awareness
        recent_facts = list(self.personal_facts.values())[-4:]
        for fact in recent_facts:
            if fact.current_status == EntityStatus.CURRENT:
                context_parts.append(f"{fact.key.replace('_', ' ')}: {fact.value}")
            else:
                context_parts.append(f"Former {fact.key.replace('_', ' ')}: {fact.value} (Status: {fact.current_status.value})")
        
        # Critical entity statuses
        critical_entities = [entity for entity in self.entity_memories.values() 
                           if entity.emotional_significance > 0.7]
        for entity in critical_entities:
            status_desc = f"{entity.name} ({entity.entity_type}): {entity.status.value}"
            if entity.status == EntityStatus.DECEASED:
                status_desc += " - Handle with sensitivity"
            context_parts.append(status_desc)
        
        # Recent emotional states with entity connections
        if self.emotional_history:
            recent_emotion = self.emotional_history[-1]
            emotion_context = f"Recent emotion: {recent_emotion.emotion} - {recent_emotion.context}"
            if recent_emotion.trigger_entities:
                emotion_context += f" (Related to: {', '.join(recent_emotion.trigger_entities)})"
            context_parts.append(emotion_context)
        
        # Active life events with ongoing effects
        recent_events = list(self.life_events.values())[-3:]
        for event in recent_events:
            if event.ongoing_effects:
                context_parts.append(f"Recent event: {event.description} - Ongoing: {', '.join(event.ongoing_effects)}")
        
        return "\n".join(context_parts) if context_parts else ""
    
    # Enhanced extraction methods with entity awareness
    def extract_memories_from_text(self, text: str):
        """🧠 MEGA-INTELLIGENT memory extraction with entity tracking"""
        try:
            text_lower = text.lower().strip()
            
            # 🧠 CRITICAL: Death and loss detection
            self._extract_death_and_loss_events(text_lower, text)
            
            # 🧠 Enhanced personal facts with entity awareness
            self._extract_enhanced_personal_facts(text_lower, text)
            
            # 🧠 Enhanced emotional states with entity connections
            self._extract_enhanced_emotional_states(text_lower, text)
            
            # 🧠 Relationship status changes
            self._extract_relationship_changes(text_lower, text)
            
            # 🧠 Enhanced events with entity connections
            self._extract_enhanced_events(text_lower, text)
            
        except Exception as e:
            if DEBUG:
                print(f"[MegaMemory] ❌ Enhanced extraction error: {e}")
    
    def _extract_death_and_loss_events(self, text_lower: str, original_text: str):
        """🧠 CRITICAL: Extract death and loss events with high emotional significance"""
        death_patterns = [
            (r"my (\w+) (died|passed away|passed|is dead|has died)", "death", "{0}", EmotionalImpact.CRITICAL.value),
            (r"(\w+) died", "death", "{0}", EmotionalImpact.CRITICAL.value),
            (r"(\w+) passed away", "death", "{0}", EmotionalImpact.CRITICAL.value),
            (r"lost my (\w+)", "loss", "{0}", EmotionalImpact.HIGH.value),
            (r"put (\w+) down", "euthanasia", "{0}", EmotionalImpact.CRITICAL.value),
            (r"had to say goodbye to (\w+)", "death", "{0}", EmotionalImpact.CRITICAL.value),
            (r"(\w+) is no longer with us", "death", "{0}", EmotionalImpact.CRITICAL.value),
        ]
        
        for pattern, event_type, name_template, emotional_impact in death_patterns:
            match = re.search(pattern, text_lower)
            if match:
                entity_name = name_template.format(match.group(1))
                
                # Determine entity type
                entity_type = self._determine_entity_type(entity_name, original_text)
                
                # Add entity memory
                self.add_entity_memory(
                    name=entity_name,
                    entity_type=entity_type,
                    status=EntityStatus.DECEASED,
                    emotional_significance=emotional_impact,
                    context=original_text
                )
                
                # Add life event
                self.add_life_event(
                    event_type=event_type,
                    description=f"{entity_name} {event_type}",
                    entities_involved=[entity_name],
                    emotional_impact=emotional_impact,
                    ongoing_effects=["grieving", "emotional_sensitivity_needed"]
                )
                
                print(f"[MegaMemory] 💔 CRITICAL EVENT: {entity_name} death/loss detected")
    
    def _determine_entity_type(self, name: str, context: str) -> str:
        """Determine entity type from context"""
        context_lower = context.lower()
        
        pet_indicators = ["cat", "dog", "pet", "puppy", "kitten", "bird", "fish", "hamster"]
        person_indicators = ["mom", "dad", "mother", "father", "friend", "sister", "brother", "grandmother", "grandfather"]
        
        if any(indicator in context_lower for indicator in pet_indicators):
            return "pet"
        elif any(indicator in context_lower for indicator in person_indicators):
            return "person"
        else:
            return "unknown"
    
    def _extract_relationship_changes(self, text_lower: str, original_text: str):
        """Extract relationship status changes"""
        relationship_patterns = [
            (r"my (ex|former) (\w+)", "relationship_end", "{1}", EntityStatus.ENDED),
            (r"broke up with (\w+)", "breakup", "{0}", EntityStatus.ENDED),
            (r"divorced (\w+)", "divorce", "{0}", EntityStatus.ENDED),
            (r"separated from (\w+)", "separation", "{0}", EntityStatus.ENDED),
        ]
        
        for pattern, event_type, name_template, status in relationship_patterns:
            match = re.search(pattern, text_lower)
            if match:
                entity_name = name_template.format(*match.groups())
                
                self.add_entity_memory(
                    name=entity_name,
                    entity_type="relationship",
                    status=status,
                    emotional_significance=EmotionalImpact.HIGH.value,
                    context=original_text
                )
                
                print(f"[MegaMemory] 💔 Relationship change: {entity_name} - {status.value}")
    
    def _extract_enhanced_personal_facts(self, text_lower: str, original_text: str):
        """Enhanced personal fact extraction with entity awareness"""
        enhanced_patterns = [
            # Physical attributes
            (r"my shoe size is (\d+)", "physical", "shoe_size", EntityStatus.CURRENT),
            (r"i'm (\d+) years old", "physical", "age", EntityStatus.CURRENT),
            
            # Preferences with entity awareness
            (r"i love my (\w+)", "preferences", "loves_{0}", EntityStatus.CURRENT),
            (r"i hate (\w+)", "preferences", "dislikes_{0}", EntityStatus.CURRENT),
            (r"i used to love (\w+)", "preferences", "formerly_loved_{0}", EntityStatus.FORMER),
            
            # Possessions with status
            (r"i have a (\w+)", "possessions", "owns_{0}", EntityStatus.CURRENT),
            (r"i used to have a (\w+)", "possessions", "formerly_owned_{0}", EntityStatus.FORMER),
            (r"i sold my (\w+)", "possessions", "sold_{0}", EntityStatus.SOLD),
            
            # Medical with ongoing status
            (r"i'm allergic to (\w+)", "medical", "allergy_{0}", EntityStatus.CURRENT),
            (r"i have (\w+) condition", "medical", "condition_{0}", EntityStatus.CURRENT),
        ]
        
        for pattern, category, key_template, status in enhanced_patterns:
            match = re.search(pattern, text_lower)
            if match:
                key = key_template.format(*match.groups()) if "{0}" in key_template else key_template
                value = match.group(1)
                
                fact = PersonalFact(
                    category=category,
                    key=key,
                    value=value,
                    confidence=0.8,
                    date_learned=datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                    last_mentioned=datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                    source_context=original_text,
                    current_status=status
                )
                
                self.personal_facts[f"{category}_{key}"] = fact
    
    def _extract_enhanced_emotional_states(self, text_lower: str, original_text: str):
        """Enhanced emotional state extraction with entity connections"""
        emotion_patterns = [
            (r"i'm (sad|depressed|down|upset) about (\w+)", "sad", 7, "{1}"),
            (r"i'm (happy|excited|thrilled) about (\w+)", "happy", 8, "{1}"),
            (r"missing (\w+)", "sad", 6, "{0}"),
            (r"grieving (\w+)", "sad", 8, "{0}"),
            (r"i miss (\w+)", "sad", 7, "{0}"),
        ]
        
        for pattern, emotion, intensity, entity_template in emotion_patterns:
            match = re.search(pattern, text_lower)
            if match:
                trigger_entity = entity_template.format(*match.groups()) if entity_template else None
                
                state = EmotionalState(
                    emotion=emotion,
                    intensity=intensity,
                    context=original_text,
                    date=datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                    follow_up_needed=True,
                    trigger_entities=[trigger_entity] if trigger_entity else []
                )
                
                self.emotional_history.append(state)
                print(f"[MegaMemory] 😢 Emotional state: {emotion} about {trigger_entity}")
    
    def _extract_enhanced_events(self, text_lower: str, original_text: str):
        """Enhanced event extraction with entity connections"""
        event_patterns = [
            (r"(?:it's|its) my (\w+)'s birthday tomorrow", "birthday", "{0}'s birthday", 1, ["{0}"]),
            (r"(\w+)'s funeral is tomorrow", "funeral", "{0}'s funeral", 1, ["{0}"]),
            (r"visiting (\w+) tomorrow", "visit", "visiting {0}", 1, ["{0}"]),
        ]
        
        for pattern, event_type, desc_template, days_ahead, entities_template in event_patterns:
            match = re.search(pattern, text_lower)
            if match:
                description = desc_template.format(*match.groups())
                entities = [template.format(*match.groups()) for template in entities_template]
                
                event_date = (datetime.datetime.utcnow() + 
                            datetime.timedelta(days=days_ahead)).strftime('%Y-%m-%d')
                
                event = ScheduledEvent(
                    event_type=event_type,
                    description=description,
                    date=event_date,
                    reminder_dates=[event_date],
                    completed=False,
                    related_entities=entities
                )
                
                self.scheduled_events.append(event)
    
    # Enhanced save/load methods
    def save_memory(self):
        """Save enhanced memory with entity awareness"""
        try:
            # Save all existing memory types
            super_save_methods = [
                self._save_personal_facts,
                self._save_emotional_history,
                self._save_scheduled_events,
                self._save_conversation_topics
            ]
            
            # 🧠 Save new enhanced memory types
            self._save_entity_memories()
            self._save_life_events()
            
            for save_method in super_save_methods:
                save_method()
                
        except Exception as e:
            if DEBUG:
                print(f"[MegaMemory] ❌ Save error: {e}")
    
    def _save_entity_memories(self):
        """Save entity memories"""
        entities_file = self.memory_dir / "entity_memories.json"
        with open(entities_file, 'w') as f:
            entities_data = {}
            for name, entity in self.entity_memories.items():
                entity_dict = asdict(entity)
                entity_dict['status'] = entity.status.value  # Convert enum to string
                entities_data[name] = entity_dict
            json.dump(entities_data, f, indent=2)
    
    def _save_life_events(self):
        """Save life events"""
        events_file = self.memory_dir / "life_events.json"
        with open(events_file, 'w') as f:
            events_data = {k: asdict(v) for k, v in self.life_events.items()}
            json.dump(events_data, f, indent=2)
    
    def load_memory(self):
        """Load enhanced memory with entity awareness"""
        try:
            # Load existing memory types
            self._load_personal_facts()
            self._load_emotional_history()
            self._load_scheduled_events()
            self._load_conversation_topics()
            
            # 🧠 Load new enhanced memory types
            self._load_entity_memories()
            self._load_life_events()
            
            if DEBUG:
                print(f"[MegaMemory] 🧠 Loaded MEGA-INTELLIGENT memory for {self.username}")
                print(f"  Entities: {len(self.entity_memories)}")
                print(f"  Life Events: {len(self.life_events)}")
                
        except Exception as e:
            if DEBUG:
                print(f"[MegaMemory] ❌ Load error: {e}")
    
    def _load_entity_memories(self):
        """Load entity memories"""
        entities_file = self.memory_dir / "entity_memories.json"
        if entities_file.exists():
            with open(entities_file, 'r') as f:
                entities_data = json.load(f)
                for name, entity_dict in entities_data.items():
                    entity_dict['status'] = EntityStatus(entity_dict['status'])  # Convert string back to enum
                    self.entity_memories[name] = EntityMemory(**entity_dict)
    
    def _load_life_events(self):
        """Load life events"""
        events_file = self.memory_dir / "life_events.json"
        if events_file.exists():
            with open(events_file, 'r') as f:
                events_data = json.load(f)
                self.life_events = {k: LifeEvent(**v) for k, v in events_data.items()}
    
    # Individual save methods for existing data types
    def _save_personal_facts(self):
        facts_file = self.memory_dir / "personal_facts.json"
        with open(facts_file, 'w') as f:
            facts_data = {}
            for k, v in self.personal_facts.items():
                fact_dict = asdict(v)
                if hasattr(v, 'current_status'):
                    fact_dict['current_status'] = v.current_status.value
                facts_data[k] = fact_dict
            json.dump(facts_data, f, indent=2)
    
    def _save_emotional_history(self):
        emotions_file = self.memory_dir / "emotions.json"
        with open(emotions_file, 'w') as f:
            emotions_data = [asdict(e) for e in self.emotional_history]
            json.dump(emotions_data, f, indent=2)
    
    def _save_scheduled_events(self):
        events_file = self.memory_dir / "events.json"
        with open(events_file, 'w') as f:
            events_data = [asdict(e) for e in self.scheduled_events]
            json.dump(events_data, f, indent=2)
    
    def _save_conversation_topics(self):
        topics_file = self.memory_dir / "conversation_topics.json"
        with open(topics_file, 'w') as f:
            topics_data = [asdict(t) for t in self.conversation_topics]
            json.dump(topics_data, f, indent=2)
    
    def _load_personal_facts(self):
        facts_file = self.memory_dir / "personal_facts.json"
        if facts_file.exists():
            with open(facts_file, 'r') as f:
                facts_data = json.load(f)
                for k, v in facts_data.items():
                    if 'current_status' in v:
                        v['current_status'] = EntityStatus(v['current_status'])
                    self.personal_facts[k] = PersonalFact(**v)
    
    def _load_emotional_history(self):
        emotions_file = self.memory_dir / "emotions.json"
        if emotions_file.exists():
            with open(emotions_file, 'r') as f:
                emotions_data = json.load(f)
                self.emotional_history = [EmotionalState(**e) for e in emotions_data]
    
    def _load_scheduled_events(self):
        events_file = self.memory_dir / "events.json"
        if events_file.exists():
            with open(events_file, 'r') as f:
                events_data = json.load(f)
                self.scheduled_events = [ScheduledEvent(**e) for e in events_data]
    
    def _load_conversation_topics(self):
        topics_file = self.memory_dir / "conversation_topics.json"
        if topics_file.exists():
            with open(topics_file, 'r') as f:
                topics_data = json.load(f)
                self.conversation_topics = [ConversationTopic(**t) for t in topics_data]
    
    # Keep existing methods for compatibility
    def add_conversation_topic(self, topic: str, keywords: List[str]):
        """Add or update a conversation topic"""
        current_time = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        
        existing_topic = None
        for t in self.conversation_topics:
            if t.topic.lower() == topic.lower():
                existing_topic = t
                break
        
        if existing_topic:
            existing_topic.last_mentioned = current_time
            existing_topic.message_count += 1
            for keyword in keywords:
                if keyword.lower() not in [k.lower() for k in existing_topic.keywords]:
                    existing_topic.keywords.append(keyword)
        else:
            new_topic = ConversationTopic(
                topic=topic,
                start_time=current_time,
                last_mentioned=current_time,
                message_count=1,
                keywords=keywords
            )
            self.conversation_topics.append(new_topic)
            
            if len(self.conversation_topics) > MAX_CONVERSATION_TOPICS:
                self.conversation_topics = self.conversation_topics[-MAX_CONVERSATION_TOPICS:]
        
        self.save_memory()
    
    def get_recent_topics(self) -> List[str]:
        """Get recently discussed topics"""
        return [topic.topic for topic in self.conversation_topics[-3:]]
    
    def add_personal_fact(self, category: str, key: str, value: str, 
                         confidence: float, context: str):
        """Add or update a personal fact"""
        fact_id = f"{category}_{key}"
        current_time = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        
        fact = PersonalFact(
            category=category,
            key=key,
            value=value,
            confidence=confidence,
            date_learned=current_time,
            last_mentioned=current_time,
            source_context=context
        )
        
        self.personal_facts[fact_id] = fact
        print(f"[MegaMemory] 📝 Learned: {self.username} {key} = {value}")
        self.save_memory()
    
    def add_emotional_state(self, emotion: str, intensity: int, 
                           context: str, follow_up: bool = True):
        """Record user's emotional state"""
        current_time = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        
        state = EmotionalState(
            emotion=emotion,
            intensity=intensity,
            context=context,
            date=current_time,
            follow_up_needed=follow_up
        )
        
        self.emotional_history.append(state)
        print(f"[MegaMemory] 😊 Emotion: {self.username} feeling {emotion} ({intensity}/10)")
        self.save_memory()
    
    def add_scheduled_event(self, event_type: str, description: str, 
                           event_date: str, reminder_days: List[int] = [1, 0]):
        """Add an event to remember"""
        event_dt = datetime.datetime.strptime(event_date, '%Y-%m-%d')
        reminder_dates = []
        
        for days_before in reminder_days:
            reminder_dt = event_dt - datetime.timedelta(days=days_before)
            reminder_dates.append(reminder_dt.strftime('%Y-%m-%d'))
        
        event = ScheduledEvent(
            event_type=event_type,
            description=description,
            date=event_date,
            reminder_dates=reminder_dates,
            completed=False
        )
        
        self.scheduled_events.append(event)
        print(f"[MegaMemory] 📅 Event: {description} on {event_date}")
        self.save_memory()
    
    def get_today_reminders(self) -> List[str]:
        """Get reminders for today"""
        today = datetime.datetime.utcnow().strftime('%Y-%m-%d')
        reminders = []
        
        for event in self.scheduled_events:
            if not event.completed and today in event.reminder_dates:
                if event.date == today:
                    reminders.append(f"Today is {event.description}!")
                else:
                    days_until = (datetime.datetime.strptime(event.date, '%Y-%m-%d') - 
                                datetime.datetime.strptime(today, '%Y-%m-%d')).days
                    reminders.append(f"{event.description} is in {days_until} day(s)")
        
        return reminders
    
    def get_follow_up_questions(self) -> List[str]:
        """Get questions to follow up on previous conversations"""
        questions = []
        today = datetime.datetime.utcnow().strftime('%Y-%m-%d')
        yesterday = (datetime.datetime.utcnow() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Check emotional follow-ups from yesterday
        for emotion in self.emotional_history[-5:]:
            emotion_date = emotion.date.split(' ')[0]
            
            if emotion.follow_up_needed and emotion_date == yesterday:
                if emotion.emotion in ["sad", "stressed", "angry", "upset"]:
                    questions.append(f"How are you feeling today? Yesterday you seemed {emotion.emotion} about {emotion.context}")
                elif emotion.emotion in ["happy", "excited", "thrilled"]:
                    questions.append(f"Are you still feeling {emotion.emotion} about {emotion.context}?")
        
        return questions[:2]
    
    def get_memory_context(self) -> str:
        """Get relevant memory context for conversation"""
        return self.get_contextual_memory_for_response()

# Global conversation storage (keep existing)
conversation_history = {}

# Global memory manager (keep existing)
user_memories: Dict[str, UserMemorySystem] = {}

def get_user_memory(username: str) -> UserMemorySystem:
    """Get or create user memory system"""
    if username not in user_memories:
        user_memories[username] = UserMemorySystem(username)
    return user_memories[username]

# Enhanced conversation functions
def add_to_conversation_history(username, user_message, ai_response):
    """🧠 Enhanced conversation history with mega-intelligent memory extraction"""
    try:
        if username not in conversation_history:
            conversation_history[username] = []
        
        conversation_history[username].append({
            "user": user_message,
            "assistant": ai_response,
            "timestamp": time.time()
        })
        
        max_length = CONVERSATION_MEMORY_LENGTH if ENHANCED_CONVERSATION_MEMORY else MAX_HISTORY_LENGTH
        if len(conversation_history[username]) > max_length:
            conversation_history[username] = conversation_history[username][-max_length:]
        
        # 🧠 MEGA-INTELLIGENT: Enhanced memory extraction
        memory = get_user_memory(username)
        
        # Extract memories from both user message and AI response
        memory.extract_memories_from_text(user_message)
        
        # Extract topic with entity awareness
        if TOPIC_TRACKING_ENABLED:
            recent_messages = [exc["user"] for exc in conversation_history[username][-2:]]
            topic = extract_topic_from_conversation(recent_messages)
            if topic != "general":
                keywords = re.findall(r'\b\w+\b', user_message.lower())
                memory.add_conversation_topic(topic, keywords[:4])
        
        if DEBUG:
            print(f"[MegaMemory] 💭 Added to MEGA-INTELLIGENT memory for {username}")
            
    except Exception as e:
        if DEBUG:
            print(f"[MegaMemory] ❌ Enhanced memory error: {e}")

def get_conversation_context(username):
    """🧠 Get MEGA-INTELLIGENT conversation context"""
    try:
        context_parts = []
        
        if username in conversation_history and conversation_history[username]:
            history = conversation_history[username]
            
            context_length = CONVERSATION_CONTEXT_LENGTH if ENHANCED_CONVERSATION_MEMORY else 2
            recent_exchanges = history[-context_length:]
            
            if (CONVERSATION_SUMMARY_ENABLED and 
                len(history) > CONVERSATION_SUMMARY_THRESHOLD):
                summary = summarize_old_conversation(history)
                if summary:
                    context_parts.append(f"Conversation summary: {summary}")
                    context_parts.append("")
            
            for exchange in recent_exchanges:
                user_msg = exchange["user"][:120]
                ai_msg = exchange["assistant"][:120]
                context_parts.append(f"Human: {user_msg}")
                context_parts.append(f"Assistant: {ai_msg}")
        
        # 🧠 MEGA-INTELLIGENT: Enhanced memory context
        memory = get_user_memory(username)
        memory_context = memory.get_contextual_memory_for_response()
        follow_ups = memory.get_follow_up_questions()
        
        if memory_context:
            context_parts.append(f"\n🧠 MEGA-INTELLIGENT Memory Context for {username}:")
            context_parts.append(memory_context)
        
        if follow_ups:
            context_parts.append(f"\nSuggested follow-up questions:")
            context_parts.extend(follow_ups)
        
        full_context = "\n".join(context_parts)
        
        if CONTEXT_COMPRESSION_ENABLED and len(full_context) > MAX_CONTEXT_TOKENS * 4:
            full_context = full_context[:MAX_CONTEXT_TOKENS * 4]
            full_context += "\n[Context trimmed for optimization]"
        
        if DEBUG and ENHANCED_CONVERSATION_MEMORY:
            print(f"[MegaMemory] 🧠 MEGA-INTELLIGENT context generated")
        
        return full_context
        
    except Exception as e:
        if DEBUG:
            print(f"[MegaMemory] ❌ Context error: {e}")
        return ""

def extract_topic_from_conversation(messages: List[str]) -> str:
    """Extract main topic from recent messages"""
    text = " ".join(messages[-2:]).lower()
    
    topic_patterns = [
        (r"\b(cat|cats|kitten|feline|meow|purr)\b", "cats"),
        (r"\b(dog|dogs|puppy|canine|bark|woof)\b", "dogs"),
        (r"\b(work|job|office|boss|colleague|meeting)\b", "work"),
        (r"\b(vacation|holiday|travel|trip|visit)\b", "vacation"),
        (r"\b(food|cooking|recipe|restaurant|eat|meal)\b", "food"),
        (r"\b(movie|film|cinema|netflix|watch)\b", "movies"),
        (r"\b(music|song|concert|band|listen)\b", "music"),
        (r"\b(family|mom|dad|sister|brother|parent|relative)\b", "family"),
        (r"\b(friend|friends|friendship|buddy)\b", "friends"),
        (r"\b(health|doctor|medical|sick|hospital)\b", "health"),
        (r"\b(shopping|buy|purchase|store|mall)\b", "shopping"),
        (r"\b(weather|rain|sun|snow|cold|hot|temperature)\b", "weather"),
        (r"\b(game|gaming|play|xbox|playstation)\b", "gaming"),
        (r"\b(book|reading|novel|author|chapter)\b", "books"),
        # 🧠 NEW: Death and loss topics
        (r"\b(died|death|passed|loss|grief|funeral)\b", "loss_and_grief"),
        (r"\b(sick|illness|medical|hospital|treatment)\b", "health_concerns"),
    ]
    
    for pattern, topic in topic_patterns:
        if re.search(pattern, text):
            return topic
    
    return "general"

def summarize_old_conversation(history: List[Dict]) -> str:
    """Create a summary of old conversation exchanges"""
    if len(history) <= CONVERSATION_SUMMARY_THRESHOLD:
        return ""
    
    old_exchanges = history[:-CONVERSATION_CONTEXT_LENGTH]
    topics = set()
    
    for exchange in old_exchanges:
        user_msg = exchange["user"].lower()
        if any(word in user_msg for word in ["cat", "cats", "kitten", "feline"]):
            topics.add("cats")
        if any(word in user_msg for word in ["work", "job", "office", "boss"]):
            topics.add("work")
        if any(word in user_msg for word in ["family", "mom", "dad", "parent"]):
            topics.add("family")
        if any(word in user_msg for word in ["vacation", "travel", "trip"]):
            topics.add("vacation")
        if any(word in user_msg for word in ["food", "cooking", "restaurant"]):
            topics.add("food")
        # 🧠 NEW: Loss and grief detection
        if any(word in user_msg for word in ["died", "death", "passed", "loss", "grief"]):
            topics.add("loss_and_grief")
    
    if topics:
        return f"Earlier we discussed: {', '.join(sorted(topics))}"
    
    return f"Earlier conversation ({len(old_exchanges)} exchanges)"

# 🧠 NEW: Response validation function
def validate_ai_response_appropriateness(username: str, proposed_response: str) -> Tuple[bool, str]:
    """🧠 MEGA-INTELLIGENT: Validate AI response before output"""
    memory = get_user_memory(username)
    return memory.validate_response_before_output(proposed_response)

print(f"[MegaMemory] 🧠 MEGA-INTELLIGENT Memory System Loaded!")
print(f"[MegaMemory] ✅ Entity Status Tracking: Active")
print(f"[MegaMemory] ✅ Life Event Detection: Active") 
print(f"[MegaMemory] ✅ Response Validation: Active")
print(f"[MegaMemory] ✅ Memory Inference Engine: Active")