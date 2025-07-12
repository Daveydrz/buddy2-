# ai/memory_fusion.py - INTELLIGENT Memory Cluster Detection & Unification
import json
import os
import shutil
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path
import re
from difflib import SequenceMatcher
from ai.memory import get_user_memory, UserMemorySystem

class MemoryClusterAnalyzer:
    """🧠 Detect when different usernames belong to the same person"""
    
    def __init__(self):
        self.memory_base_dir = Path("memory")
        self.cluster_file = self.memory_base_dir / "user_clusters.json"
        self.load_existing_clusters()
    
    def load_existing_clusters(self):
        """Load known user clusters"""
        if self.cluster_file.exists():
            with open(self.cluster_file, 'r') as f:
                self.clusters = json.load(f)
        else:
            self.clusters = {}
    
    def save_clusters(self):
        """Save user clusters"""
        os.makedirs(self.memory_base_dir, exist_ok=True)
        with open(self.cluster_file, 'w') as f:
            json.dump(self.clusters, f, indent=2)
    
    def analyze_user_similarity(self, user1: str, user2: str) -> float:
        """Calculate similarity between two users based on multiple factors"""
        
        similarity_factors = []
        
        # 1. Username pattern similarity (Anonymous_001, Anonymous_002 = high similarity)
        username_sim = self._calculate_username_similarity(user1, user2)
        similarity_factors.append(("username", username_sim, 0.3))
        
        # 2. Personal facts overlap (husband=francesco, likes dogs, etc.)
        facts_sim = self._calculate_facts_similarity(user1, user2)
        similarity_factors.append(("facts", facts_sim, 0.5))  # High weight for facts
        
        # 3. Conversation patterns
        conversation_sim = self._calculate_conversation_similarity(user1, user2)
        similarity_factors.append(("conversation", conversation_sim, 0.2))
        
        # Calculate weighted average
        total_weight = sum(weight for _, _, weight in similarity_factors)
        weighted_score = sum(score * weight for _, score, weight in similarity_factors) / total_weight
        
        if weighted_score > 0.3:  # Only print if somewhat similar
            print(f"[MemoryFusion] 📊 {user1} ↔ {user2} similarity: {weighted_score:.2f}")
            for factor_name, score, weight in similarity_factors:
                print(f"  {factor_name}: {score:.2f} (weight: {weight})")
        
        return weighted_score
    
    def _calculate_username_similarity(self, user1: str, user2: str) -> float:
        """Detect username patterns (Anonymous_001, Anonymous_002, etc.)"""
        
        # Check for Anonymous pattern
        anon_pattern = r'^Anonymous_\d{3}$'
        if re.match(anon_pattern, user1) and re.match(anon_pattern, user2):
            return 0.9  # Very high similarity for Anonymous users
        
        # Check if one is Anonymous and other is real name (likely same person)
        if (re.match(anon_pattern, user1) and not re.match(anon_pattern, user2)) or \
           (re.match(anon_pattern, user2) and not re.match(anon_pattern, user1)):
            return 0.6  # Medium similarity - could be same person revealing real name
        
        # General string similarity
        return SequenceMatcher(None, user1.lower(), user2.lower()).ratio()
    
    def _calculate_facts_similarity(self, user1: str, user2: str) -> float:
        """Compare personal facts between users"""
        
        try:
            facts1 = self._load_user_facts(user1)
            facts2 = self._load_user_facts(user2)
            
            if not facts1 or not facts2:
                return 0.0
            
            # Look for exact matches in values
            exact_matches = 0
            total_comparisons = 0
            
            for key1, value1 in facts1.items():
                for key2, value2 in facts2.items():
                    total_comparisons += 1
                    if value1.lower().strip() == value2.lower().strip():
                        exact_matches += 1
                        print(f"[MemoryFusion] 🎯 EXACT MATCH: {key1}='{value1}' ↔ {key2}='{value2}'")
            
            if total_comparisons == 0:
                return 0.0
            
            similarity = exact_matches / total_comparisons
            
            # Boost for critical matches (relationships, names)
            critical_matches = self._find_critical_fact_matches(facts1, facts2)
            if critical_matches > 0:
                similarity += critical_matches * 0.4  # Each critical match adds 40%
                print(f"[MemoryFusion] 🔥 Found {critical_matches} critical matches!")
            
            return min(similarity, 1.0)
            
        except Exception as e:
            print(f"[MemoryFusion] ⚠️ Facts comparison error: {e}")
            return 0.0
    
    def _load_user_facts(self, username: str) -> Dict[str, str]:
        """Load personal facts for a user"""
        facts_file = Path(f"memory/{username}/personal_facts.json")
        if not facts_file.exists():
            return {}
        
        try:
            with open(facts_file, 'r') as f:
                facts_data = json.load(f)
            
            # Extract key-value pairs from your MEGA-INTELLIGENT format
            facts = {}
            for key, fact_obj in facts_data.items():
                if isinstance(fact_obj, dict) and 'value' in fact_obj:
                    clean_key = fact_obj.get('key', key).replace('_', ' ')
                    facts[clean_key] = fact_obj['value']
                else:
                    facts[key] = str(fact_obj)
            
            return facts
            
        except Exception as e:
            print(f"[MemoryFusion] ⚠️ Error loading facts for {username}: {e}")
            return {}
    
    def _find_critical_fact_matches(self, facts1: Dict, facts2: Dict) -> int:
        """Find matches in critical facts (names, relationships, etc.)"""
        
        critical_keywords = [
            'husband', 'wife', 'spouse', 'partner', 'name', 'francesco', 
            'dog', 'cat', 'pet', 'work', 'job', 'loves', 'likes'
        ]
        
        matches = 0
        
        for key1, value1 in facts1.items():
            for key2, value2 in facts2.items():
                # Check if values match and involve critical keywords
                if value1.lower() == value2.lower():
                    for keyword in critical_keywords:
                        if (keyword in key1.lower() or keyword in value1.lower() or
                            keyword in key2.lower() or keyword in value2.lower()):
                            matches += 1
                            print(f"[MemoryFusion] 🎯 CRITICAL: {key1}='{value1}' ↔ {key2}='{value2}'")
                            break
        
        return matches
    
    def _calculate_conversation_similarity(self, user1: str, user2: str) -> float:
        """Compare conversation patterns and topics"""
        
        try:
            topics1 = self._load_conversation_topics(user1)
            topics2 = self._load_conversation_topics(user2)
            
            if not topics1 or not topics2:
                return 0.0
            
            # Compare topics
            common_topics = set(topics1) & set(topics2)
            total_topics = set(topics1) | set(topics2)
            
            if len(total_topics) == 0:
                return 0.0
            
            similarity = len(common_topics) / len(total_topics)
            print(f"[MemoryFusion] 💬 Common topics: {common_topics}")
            return similarity
            
        except Exception:
            return 0.0
    
    def _load_conversation_topics(self, username: str) -> List[str]:
        """Load conversation topics for a user"""
        topics_file = Path(f"memory/{username}/conversation_topics.json")
        if not topics_file.exists():
            return []
        
        try:
            with open(topics_file, 'r') as f:
                topics_data = json.load(f)
            
            topics = []
            for topic_obj in topics_data:
                if isinstance(topic_obj, dict) and 'topic' in topic_obj:
                    topics.append(topic_obj['topic'])
                elif isinstance(topic_obj, str):
                    topics.append(topic_obj)
            
            return topics
            
        except Exception:
            return []

class MemoryUnificationEngine:
    """🧠 Merge memories from multiple usernames into unified identity"""
    
    def __init__(self):
        self.analyzer = MemoryClusterAnalyzer()
    
    def find_similar_users(self, target_username: str, threshold: float = 0.6) -> List[Tuple[str, float]]:
        """Find all users similar to target username"""
        
        if not os.path.exists(f"memory/{target_username}"):
            return []
        
        # Get all existing usernames
        existing_users = [d for d in os.listdir("memory") 
                         if os.path.isdir(f"memory/{d}") and d != target_username]
        
        similar_users = []
        
        for existing_user in existing_users:
            similarity = self.analyzer.analyze_user_similarity(target_username, existing_user)
            if similarity >= threshold:
                similar_users.append((existing_user, similarity))
                print(f"[MemoryFusion] 🎯 SIMILAR USER: {existing_user} (similarity: {similarity:.2f})")
        
        # Sort by similarity (highest first)
        similar_users.sort(key=lambda x: x[1], reverse=True)
        return similar_users
    
    def unify_memories(self, primary_username: str, secondary_usernames: List[str]) -> bool:
        """Merge memories from secondary usernames into primary username"""
        
        try:
            print(f"[MemoryFusion] 🔄 Starting memory unification:")
            print(f"  Primary: {primary_username}")
            print(f"  Secondary: {', '.join(secondary_usernames)}")
            
            # Create backup
            self._create_backup(primary_username, secondary_usernames)
            
            # Load primary memory system
            primary_memory = get_user_memory(primary_username)
            
            # Merge each secondary memory
            merged_count = 0
            for secondary_username in secondary_usernames:
                if self._merge_user_memory(primary_memory, secondary_username):
                    merged_count += 1
            
            # Save unified memory
            primary_memory.save_memory()
            
            # Update cluster mapping
            self._update_cluster_mapping(primary_username, secondary_usernames)
            
            print(f"[MemoryFusion] ✅ Successfully unified {merged_count} memories into {primary_username}")
            return True
            
        except Exception as e:
            print(f"[MemoryFusion] ❌ Unification error: {e}")
            return False
    
    def _create_backup(self, primary_username: str, secondary_usernames: List[str]):
        """Create backup of all memories before merging"""
        
        backup_dir = Path(f"memory_backups/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup all users
        all_users = [primary_username] + secondary_usernames
        for username in all_users:
            if os.path.exists(f"memory/{username}"):
                shutil.copytree(f"memory/{username}", backup_dir / username)
        
        print(f"[MemoryFusion] 💾 Backup created: {backup_dir}")
    
    def _merge_user_memory(self, primary_memory: UserMemorySystem, secondary_username: str) -> bool:
        """Merge a secondary user's memory into primary"""
        
        try:
            if not os.path.exists(f"memory/{secondary_username}"):
                return False
            
            secondary_memory = get_user_memory(secondary_username)
            
            # Merge personal facts (avoid duplicates)
            for key, fact in secondary_memory.personal_facts.items():
                if key not in primary_memory.personal_facts:
                    primary_memory.personal_facts[key] = fact
                    print(f"[MemoryFusion] ➕ Added fact: {key} = {fact.value}")
            
            # Merge emotional history
            primary_memory.emotional_history.extend(secondary_memory.emotional_history)
            
            # Merge scheduled events
            primary_memory.scheduled_events.extend(secondary_memory.scheduled_events)
            
            # Merge conversation topics
            primary_memory.conversation_topics.extend(secondary_memory.conversation_topics)
            
            # Merge entity memories (MEGA-INTELLIGENT system)
            if hasattr(secondary_memory, 'entity_memories'):
                for name, entity in secondary_memory.entity_memories.items():
                    if name not in primary_memory.entity_memories:
                        primary_memory.entity_memories[name] = entity
                        print(f"[MemoryFusion] ➕ Added entity: {name} ({entity.entity_type})")
            
            # Merge life events
            if hasattr(secondary_memory, 'life_events'):
                primary_memory.life_events.update(secondary_memory.life_events)
            
            print(f"[MemoryFusion] ✅ Merged {secondary_username} into {primary_memory.username}")
            return True
            
        except Exception as e:
            print(f"[MemoryFusion] ❌ Error merging {secondary_username}: {e}")
            return False
    
    def _update_cluster_mapping(self, primary_username: str, secondary_usernames: List[str]):
        """Update cluster mapping to track unified identities"""
        
        cluster_id = f"cluster_{primary_username}"
        self.analyzer.clusters[cluster_id] = {
            "primary_username": primary_username,
            "secondary_usernames": secondary_usernames,
            "unified_date": datetime.now().isoformat()
        }
        
        # Add mapping for quick lookup
        for username in [primary_username] + secondary_usernames:
            self.analyzer.clusters[f"mapping_{username}"] = cluster_id
        
        self.analyzer.save_clusters()
        print(f"[MemoryFusion] 🗂️ Cluster mapping saved for {primary_username}")

# Global memory fusion engine
memory_fusion_engine = MemoryUnificationEngine()

def auto_detect_and_unify_memories(current_username: str) -> str:
    """Automatically detect and unify similar user memories"""
    
    print(f"[MemoryFusion] 🔍 Checking for similar users to {current_username}")
    
    # Check if already mapped
    mapping_key = f"mapping_{current_username}"
    if mapping_key in memory_fusion_engine.analyzer.clusters:
        cluster_id = memory_fusion_engine.analyzer.clusters[mapping_key]
        if cluster_id in memory_fusion_engine.analyzer.clusters:
            primary = memory_fusion_engine.analyzer.clusters[cluster_id]["primary_username"]
            print(f"[MemoryFusion] 🔗 {current_username} already mapped to {primary}")
            return primary
    
    # Find similar users
    similar_users = memory_fusion_engine.find_similar_users(current_username, threshold=0.5)
    
    if similar_users:
        print(f"[MemoryFusion] 🎯 Found {len(similar_users)} similar users!")
        
        # Get usernames to merge
        usernames_to_merge = [user for user, similarity in similar_users]
        
        # Unify memories
        if memory_fusion_engine.unify_memories(current_username, usernames_to_merge):
            print(f"[MemoryFusion] 🚀 Memory unification complete for {current_username}!")
            return current_username
    else:
        print(f"[MemoryFusion] ℹ️ No similar users found for {current_username}")
    
    return current_username

def get_unified_username(original_username: str) -> str:
    """Get the unified username for memory operations"""
    return auto_detect_and_unify_memories(original_username)