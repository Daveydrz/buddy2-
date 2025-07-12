# voice/manager.py - INTELLIGENT Voice Learning Manager
import time
import numpy as np
from datetime import datetime, timedelta
from voice.database import known_users, save_known_users, load_known_users, anonymous_clusters
from voice.recognition import identify_speaker_with_confidence, generate_voice_embedding
from config import DEBUG
from audio.output import speak_streaming

# At the top of manager.py, after the imports
from voice.database import known_users, anonymous_clusters, save_known_users

# ADD THESE DEBUG LINES TO CHECK DICTIONARY IDs
print(f"[VoiceManager] 🔍 Using known_users id: {id(known_users)}")
print(f"[VoiceManager] 🔍 Using anonymous_clusters id: {id(anonymous_clusters)}")

# FORCE GLOBAL REFERENCE
import voice.database as db
known_users = db.known_users
anonymous_clusters = db.anonymous_clusters

class IntelligentVoiceManager:
    """🧠 INTELLIGENT Voice Learning Manager - Learns and adapts to voice changes"""
    
    def __init__(self):

        global known_users, anonymous_clusters
        from voice.database import known_users as db_known_users, anonymous_clusters as db_anonymous_clusters
        known_users = db_known_users
        anonymous_clusters = db_anonymous_clusters
    
        print(f"[VoiceManager] 🔍 Synced to database dictionaries:")
        print(f"[VoiceManager] 📊 known_users id: {id(known_users)}")
        print(f"[VoiceManager] 📊 anonymous_clusters id: {id(anonymous_clusters)}")

        self.current_user = "Daveydrz"
        self.session_start = datetime.utcnow()
        self.interactions = 0
        self.waiting_for_name = False
        self.pending_name_confirmation = False
        self.suggested_name = None
        
        # 🧠 INTELLIGENT VOICE LEARNING
        self.current_speaker_cluster_id = None
        self.current_voice_embedding = None
        self.voice_learning_history = {}  # Track voice patterns over time
        self.uncertainty_threshold = 0.6  # Ask for confirmation below this
        self.similarity_threshold = 0.65   # Cluster together above this
        self.learning_rate = 0.1  # How fast to adapt to voice changes
        
        # 🎯 SMART VOICE CLUSTERING
        self.pending_confirmation_cluster = None
        self.pending_confirmation_name = None
        self.waiting_for_voice_confirmation = False

        self.last_audio_buffer = None
        self.last_audio_timestamp = None
        self.last_identified_user = None
        
        # ✅ CRITICAL: Load existing database
        load_known_users()
        print(f"[IntelligentVoiceManager] 🧠 Intelligent voice learning initialized")
        print(f"[IntelligentVoiceManager] 📚 Loaded {len(known_users)} voice profiles")
        print(f"[IntelligentVoiceManager] 🔍 Anonymous clusters: {len(anonymous_clusters)}")
    
    def handle_voice_identification(self, audio, text):
        """🧠 INTELLIGENT voice identification with learning and adaptation"""
        try:
            self.interactions += 1
            
            # ✅ CRITICAL: Always try to save interaction data
            self._log_interaction(audio, text)
            
            # ✅ Generate current voice embedding
            current_embedding = self._generate_current_embedding(audio)
            if current_embedding is None:
                return "Daveydrz", "NO_EMBEDDING"
            
            self.current_voice_embedding = current_embedding
            
            # Handle voice confirmation flow
            if self.waiting_for_voice_confirmation:
                return self._handle_voice_confirmation(text)
            
            # Handle name confirmation flow
            if self.pending_name_confirmation:
                return self._handle_name_confirmation(text)
            
            if self.waiting_for_name:
                return self._handle_name_waiting(text)
            
            # 🧠 STEP 1: Try precise voice recognition
            identified_user, confidence = identify_speaker_with_confidence(audio)
            
            if identified_user != "UNKNOWN" and confidence >= 0.4:
                print(f"[IntelligentVoiceManager] ✅ HIGH CONFIDENCE: {identified_user} ({confidence:.3f})")
                
                # Add to learning history
                self._update_voice_learning_history(identified_user, current_embedding, confidence)
                self.set_current_cluster(identified_user)
                return identified_user, "HIGH_CONFIDENCE_RECOGNIZED"
            
            # 🧠 STEP 2: Smart similarity search across ALL voice data
            best_match = self._find_best_voice_match(current_embedding)
            
            if best_match:
                match_id, similarity, match_type = best_match
                
                # 🎯 HIGH SIMILARITY - Confident match
                if similarity >= 0.75:
                    print(f"[IntelligentVoiceManager] 🧠 SMART MATCH: {match_id} (similarity: {similarity:.3f})")
                    
                    # Add embedding to existing profile
                    self._add_embedding_to_profile(match_id, current_embedding)
                    self._update_voice_learning_history(match_id, current_embedding, similarity)
                    self.set_current_cluster(match_id)
                    
                    return match_id, "SMART_VOICE_MATCH"
                
                # 🤔 MEDIUM SIMILARITY - Ask for confirmation
                elif similarity >= self.uncertainty_threshold:
                    print(f"[IntelligentVoiceManager] 🤔 UNCERTAIN MATCH: {match_id} (similarity: {similarity:.3f})")
                    
                    # Ask user for confirmation
                    return self._ask_for_voice_confirmation(match_id, current_embedding, text)
                
                # 🆕 LOW SIMILARITY - Create new cluster but track relationship
                else:
                    print(f"[IntelligentVoiceManager] 🆕 NEW VOICE: Creating cluster (best match: {match_id}, similarity: {similarity:.3f})")
                    return self._create_new_cluster_with_tracking(current_embedding, best_match)
            
            # 🆕 NO MATCHES - Completely new voice
            else:
                print(f"[IntelligentVoiceManager] 🆕 COMPLETELY NEW VOICE: Creating first cluster")
                return self._create_new_cluster_with_tracking(current_embedding, None)
                
        except Exception as e:
            print(f"[IntelligentVoiceManager] ❌ Error: {e}")
            return "Daveydrz", "ERROR"

    def get_last_audio_sample(self):
        """Get the most recent audio sample for voice identification"""
        try:
            if hasattr(self, 'last_audio_buffer') and self.last_audio_buffer is not None:
                return self.last_audio_buffer
            elif hasattr(self, 'passive_samples') and self.passive_samples:
                return self.passive_samples[-1]['audio']  # Get most recent
            else:
                return None
        except Exception as e:
            print(f"[VoiceManager] ⚠️ Error getting last audio: {e}")
            return None

    def get_current_speaker_identity(self):
        """Get current speaker identity from advanced voice processing"""
        try:
            if hasattr(self, 'current_training_user') and self.current_training_user:
                return self.current_training_user
            elif hasattr(self, 'last_identified_user') and self.last_identified_user:
                return self.last_identified_user
            else:
                return None
        except Exception as e:
            print(f"[VoiceManager] ⚠️ Error getting speaker identity: {e}")
            return None

    def set_last_audio_sample(self, audio_data):
        """Store the most recent audio sample"""
        try:
            self.last_audio_buffer = audio_data
            # Also update timestamp
            import time
            self.last_audio_timestamp = time.time()
        except Exception as e:
            print(f"[VoiceManager] ⚠️ Error storing audio sample: {e}")

    def is_llm_locked(self):
        """Check if LLM should be locked due to voice processing"""
        try:
            # Check if voice training is in progress
            if hasattr(self, 'pending_training_offer') and self.pending_training_offer:
                return True
        
            # Check if name collection is in progress
            if hasattr(self, 'waiting_for_name') and self.waiting_for_name:
                return True
            
            # Check if any voice processing state is active
            if hasattr(self, 'training_mode') and self.training_mode != "NONE":
                return True
            
            return False
        except Exception as e:
            print(f"[VoiceManager] ⚠️ Error checking LLM lock: {e}")
    
    def _find_best_voice_match(self, current_embedding):
        """🔍 INTELLIGENT search for best voice match across all profiles"""
        try:
            best_match = None
            best_similarity = 0.0
            
            # Search through all known users
            for user_id, user_data in known_users.items():
                if isinstance(user_data, dict):
                    embeddings = user_data.get('embeddings', [])
                    
                    for stored_embedding in embeddings:
                        similarity = self._calculate_voice_similarity(current_embedding, stored_embedding)
                        
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = (user_id, similarity, 'known_user')
            
            # Search through anonymous clusters
            for cluster_id, cluster_data in anonymous_clusters.items():
                embeddings = cluster_data.get('embeddings', [])
                
                for stored_embedding in embeddings:
                    similarity = self._calculate_voice_similarity(current_embedding, stored_embedding)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = (cluster_id, similarity, 'anonymous_cluster')
            
            return best_match
            
        except Exception as e:
            print(f"[IntelligentVoiceManager] ❌ Error finding voice match: {e}")
            return None
    
    def _calculate_voice_similarity(self, embedding1, embedding2):
        """🧮 Calculate cosine similarity between voice embeddings"""
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Normalize vectors
            norm1 = vec1 / (np.linalg.norm(vec1) + 1e-8)
            norm2 = vec2 / (np.linalg.norm(vec2) + 1e-8)
            
            # Calculate cosine similarity
            similarity = np.dot(norm1, norm2)
            
            return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
            
        except Exception as e:
            print(f"[IntelligentVoiceManager] ❌ Similarity calculation error: {e}")
            return 0.0
    
    def _ask_for_voice_confirmation(self, match_id, current_embedding, text):
        """❓ Ask user to confirm if this is them"""
        try:
            # Store pending confirmation data
            self.pending_confirmation_cluster = match_id
            self.pending_confirmation_embedding = current_embedding
            self.waiting_for_voice_confirmation = True
            
            # Get display name
            display_name = self._get_display_name(match_id)
            
            # Ask for confirmation
            speak_streaming(f"Is this {display_name}?")
            print(f"[IntelligentVoiceManager] ❓ ASKING CONFIRMATION: Is this {display_name}?")
            
            return match_id, "ASKING_VOICE_CONFIRMATION"
            
        except Exception as e:
            print(f"[IntelligentVoiceManager] ❌ Error asking confirmation: {e}")
            return self._create_new_cluster_with_tracking(current_embedding, None)
    
    def _handle_voice_confirmation(self, text):
        """✅ Handle voice confirmation response"""
        try:
            text_lower = text.lower().strip()
            
            if any(word in text_lower for word in ["yes", "yeah", "yep", "correct", "right"]):
                # ✅ CONFIRMED - Add embedding to profile
                match_id = self.pending_confirmation_cluster
                current_embedding = self.pending_confirmation_embedding
                
                print(f"[IntelligentVoiceManager] ✅ VOICE CONFIRMED: {match_id}")
                
                # Add embedding and update learning history
                self._add_embedding_to_profile(match_id, current_embedding)
                self._update_voice_learning_history(match_id, current_embedding, 0.8)
                
                # Reset confirmation state
                self._reset_confirmation_state()
                self.set_current_cluster(match_id)
                
                return match_id, "VOICE_CONFIRMED"
                
            elif any(word in text_lower for word in ["no", "nope", "wrong", "not"]):
                # ❌ REJECTED - Create new cluster
                print(f"[IntelligentVoiceManager] ❌ VOICE REJECTED: Creating new cluster")
                
                current_embedding = self.pending_confirmation_embedding
                self._reset_confirmation_state()
                
                return self._create_new_cluster_with_tracking(current_embedding, None)
                
            else:
                # 🤔 UNCLEAR RESPONSE - Ask again
                display_name = self._get_display_name(self.pending_confirmation_cluster)
                speak_streaming(f"Please say yes or no. Is this {display_name}?")
                return self.pending_confirmation_cluster, "ASKING_VOICE_CONFIRMATION"
                
        except Exception as e:
            print(f"[IntelligentVoiceManager] ❌ Confirmation error: {e}")
            self._reset_confirmation_state()
            return "Daveydrz", "CONFIRMATION_ERROR"
    
    def _add_embedding_to_profile(self, profile_id, embedding):
        """📈 Add embedding to existing profile with smart learning"""
        try:
            # Handle known users
            if profile_id in known_users:
                user_data = known_users[profile_id]
                embeddings = user_data.get('embeddings', [])
                embeddings.append(embedding.tolist())
                
                # Smart pruning - keep diverse samples
                embeddings = self._smart_prune_embeddings(embeddings, max_samples=20)
                
                user_data['embeddings'] = embeddings
                user_data['last_updated'] = datetime.utcnow().isoformat()
                user_data['sample_count'] = len(embeddings)
                
                save_known_users()
                print(f"[IntelligentVoiceManager] 📈 Added embedding to known user {profile_id} (total: {len(embeddings)})")
                
            # Handle anonymous clusters
            elif profile_id in anonymous_clusters:
                cluster_data = anonymous_clusters[profile_id]
                embeddings = cluster_data.get('embeddings', [])
                embeddings.append(embedding.tolist())
                
                # Smart pruning
                embeddings = self._smart_prune_embeddings(embeddings, max_samples=15)
                
                cluster_data['embeddings'] = embeddings
                cluster_data['last_updated'] = datetime.utcnow().isoformat()
                cluster_data['sample_count'] = len(embeddings)
                
                save_known_users()
                print(f"[IntelligentVoiceManager] 📈 Added embedding to anonymous cluster {profile_id} (total: {len(embeddings)})")
                
        except Exception as e:
            print(f"[IntelligentVoiceManager] ❌ Error adding embedding: {e}")
    
    def _smart_prune_embeddings(self, embeddings, max_samples=20):
        """🧠 SMART pruning - keep diverse voice samples"""
        try:
            if len(embeddings) <= max_samples:
                return embeddings
            
            # Convert to numpy for processing
            embedding_arrays = [np.array(emb) for emb in embeddings]
            
            # Keep the most recent samples
            recent_samples = embedding_arrays[-max_samples//2:]
            
            # Keep diverse older samples
            older_samples = embedding_arrays[:-max_samples//2]
            if older_samples:
                # Select diverse samples using simple distance-based selection
                selected_older = self._select_diverse_samples(older_samples, max_samples//2)
                final_samples = selected_older + recent_samples
            else:
                final_samples = recent_samples
            
            # Convert back to lists
            return [emb.tolist() for emb in final_samples]
            
        except Exception as e:
            print(f"[IntelligentVoiceManager] ❌ Pruning error: {e}")
            # Fallback: keep most recent samples
            return embeddings[-max_samples:]
    
    def _select_diverse_samples(self, embeddings, num_samples):
        """🎯 Select diverse voice samples to maintain variety"""
        try:
            if len(embeddings) <= num_samples:
                return embeddings
            
            # Simple diversity selection: evenly spaced samples
            indices = np.linspace(0, len(embeddings)-1, num_samples, dtype=int)
            return [embeddings[i] for i in indices]
            
        except Exception as e:
            print(f"[IntelligentVoiceManager] ❌ Diversity selection error: {e}")
            return embeddings[:num_samples]
    
    def _update_voice_learning_history(self, profile_id, embedding, confidence):
        """📚 Update voice learning history for adaptation"""
        try:
            if profile_id not in self.voice_learning_history:
                self.voice_learning_history[profile_id] = {
                    'samples': [],
                    'avg_confidence': 0.0,
                    'voice_stability': 1.0,
                    'last_seen': datetime.utcnow().isoformat()
                }
            
            history = self.voice_learning_history[profile_id]
            
            # Add sample
            history['samples'].append({
                'embedding': embedding.tolist(),
                'confidence': confidence,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            # Keep only recent samples
            if len(history['samples']) > 50:
                history['samples'] = history['samples'][-50:]
            
            # Update statistics
            confidences = [s['confidence'] for s in history['samples']]
            history['avg_confidence'] = sum(confidences) / len(confidences)
            history['last_seen'] = datetime.utcnow().isoformat()
            
            # Calculate voice stability (how consistent the voice is)
            if len(history['samples']) >= 3:
                recent_embeddings = [np.array(s['embedding']) for s in history['samples'][-5:]]
                stability = self._calculate_voice_stability(recent_embeddings)
                history['voice_stability'] = stability
            
            print(f"[IntelligentVoiceManager] 📚 Updated learning history for {profile_id}: confidence={history['avg_confidence']:.3f}, stability={history['voice_stability']:.3f}")
            
        except Exception as e:
            print(f"[IntelligentVoiceManager] ❌ Learning history error: {e}")
    
    def _calculate_voice_stability(self, embeddings):
        """📊 Calculate how stable/consistent a voice is"""
        try:
            if len(embeddings) < 2:
                return 1.0
            
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    sim = self._calculate_voice_similarity(embeddings[i], embeddings[j])
                    similarities.append(sim)
            
            # Average similarity indicates stability
            avg_similarity = sum(similarities) / len(similarities)
            return avg_similarity
            
        except Exception as e:
            print(f"[IntelligentVoiceManager] ❌ Stability calculation error: {e}")
            return 0.5
    
    def _create_new_cluster_with_tracking(self, embedding, related_match=None):
        """🆕 Create new cluster with intelligent tracking"""
        try:
            cluster_id = f"Anonymous_{len(anonymous_clusters) + 1:03d}"
            
            cluster_data = {
                'cluster_id': cluster_id,
                'embeddings': [embedding.tolist()],
                'created_at': datetime.utcnow().isoformat(),
                'last_updated': datetime.utcnow().isoformat(),
                'sample_count': 1,
                'status': 'anonymous',
                'related_to': related_match[0] if related_match else None,
                'similarity_to_related': related_match[1] if related_match else None
            }
            
            anonymous_clusters[cluster_id] = cluster_data
            save_known_users()
            
            self.set_current_cluster(cluster_id)
            print(f"[IntelligentVoiceManager] 🆕 Created new cluster: {cluster_id}")
            
            if related_match:
                print(f"[IntelligentVoiceManager] 🔗 Related to: {related_match[0]} (similarity: {related_match[1]:.3f})")
            
            return cluster_id, "NEW_CLUSTER_CREATED"
            
        except Exception as e:
            print(f"[IntelligentVoiceManager] ❌ Cluster creation error: {e}")
            return "Daveydrz", "CLUSTER_CREATION_ERROR"
    
    def _get_display_name(self, profile_id):
        """📛 Get display name for profile"""
        if profile_id in known_users:
            user_data = known_users[profile_id]
            return user_data.get('name', profile_id)
        elif profile_id.startswith('Anonymous_'):
            return f"Speaker {profile_id.split('_')[1]}"
        else:
            return profile_id
    
    def _reset_confirmation_state(self):
        """🔄 Reset voice confirmation state"""
        self.waiting_for_voice_confirmation = False
        self.pending_confirmation_cluster = None
        self.pending_confirmation_embedding = None
    
    def _generate_current_embedding(self, audio):
        """🎤 Generate embedding from current audio"""
        try:
            embedding = generate_voice_embedding(audio)
            return embedding
        except Exception as e:
            print(f"[IntelligentVoiceManager] ❌ Embedding generation error: {e}")
            return None
    
    # ========== EXISTING METHODS (Keep your existing methods) ==========
    
    def _log_interaction(self, audio, text):
        """Log every interaction for debugging"""
        try:
            log_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'text': text,
                'audio_length': len(audio) if audio is not None else 0,
                'interaction_id': self.interactions
            }
            
            # Save to debug log
            import json
            debug_file = "voice_interactions_debug.json"
            
            try:
                with open(debug_file, 'r') as f:
                    logs = json.load(f)
            except:
                logs = []
            
            logs.append(log_entry)
            
            # Keep only last 50 interactions
            if len(logs) > 50:
                logs = logs[-50:]
            
            with open(debug_file, 'w') as f:
                json.dump(logs, f, indent=2)
            
            print(f"[IntelligentVoiceManager] 📝 Logged interaction #{self.interactions}")
            
        except Exception as e:
            print(f"[IntelligentVoiceManager] ❌ Logging error: {e}")
    
    def _extract_name_from_text(self, text):
        """Extract name from speech"""
        import re
        
        patterns = [
            r"my name is (\w+)",
            r"i'm (\w+)",
            r"i am (\w+)",
            r"call me (\w+)",
            r"this is (\w+)",
            r"it's (\w+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                name = match.group(1).title()
                if len(name) >= 2 and name.isalpha():
                    return name
        return None
    
    def _handle_name_introduction(self, name, audio, text):
        """Handle when user introduces themselves"""
        print(f"[IntelligentVoiceManager] 🎭 Name introduction detected: {name}")
        
        # ✅ CRITICAL: Create voice profile immediately
        success = self._create_voice_profile(name, audio)
        
        if success:
            print(f"[IntelligentVoiceManager] ✅ Created profile for: {name}")
            
            # ✅ CRITICAL: Link current cluster to this name
            if hasattr(self, 'current_speaker_cluster_id') and self.current_speaker_cluster_id:
                cluster_id = self.current_speaker_cluster_id
                if cluster_id.startswith('Anonymous_'):
                    self.update_current_speaker_name(name)
            
            return name, "NAME_CONFIRMED"
        else:
            print(f"[IntelligentVoiceManager] ❌ Failed to create profile for: {name}")
            return "Daveydrz", "PROFILE_CREATION_FAILED"
    
    def _handle_name_confirmation(self, text):
        """Handle name confirmation"""
        text_lower = text.lower().strip()
        
        if any(word in text_lower for word in ["yes", "yeah", "correct", "right", "ok"]):
            print(f"[IntelligentVoiceManager] ✅ Name confirmed: {self.suggested_name}")
            self.pending_name_confirmation = False
            confirmed_name = self.suggested_name
            self.suggested_name = None
            return confirmed_name, "NAME_CONFIRMED"
        else:
            print(f"[IntelligentVoiceManager] ❌ Name rejected")
            self.pending_name_confirmation = False
            self.waiting_for_name = True
            return "Guest", "WAITING_FOR_NAME"
    
    def _handle_name_waiting(self, text):
        """Handle waiting for name"""
        name = self._extract_name_from_text(text)
        if name:
            print(f"[IntelligentVoiceManager] 👤 Name extracted: {name}")
            self.waiting_for_name = False
            self.pending_name_confirmation = True
            self.suggested_name = name
            return "Guest", "CONFIRMING_NAME"
        else:
            print(f"[IntelligentVoiceManager] ❓ No valid name found in: '{text}'")
            return "Guest", "WAITING_FOR_NAME"
    
    def _create_voice_profile(self, username, audio):
        """Create a basic voice profile"""
        try:
            from voice.recognition import generate_voice_embedding
            
            # Generate embedding
            embedding = generate_voice_embedding(audio)
            if embedding is None:
                print(f"[IntelligentVoiceManager] ❌ Failed to generate embedding for {username}")
                return False
            
            # ✅ CRITICAL: Create profile data
            profile_data = {
                'username': username,
                'name': username,
                'embeddings': [embedding.tolist()],
                'created_date': datetime.utcnow().isoformat(),
                'confidence_threshold': 0.4,
                'status': 'intelligent_trained',
                'recognition_count': 0,
                'last_updated': datetime.utcnow().isoformat(),
                'training_type': 'intelligent_introduction',
                'system': 'IntelligentVoiceManager'
            }
            
            # ✅ CRITICAL: Save to database
            known_users[username] = profile_data
            save_known_users()
            
            print(f"[IntelligentVoiceManager] 💾 Saved profile for: {username}")
            print(f"[IntelligentVoiceManager] 📊 Total profiles: {len(known_users)}")
            
            return True
            
        except Exception as e:
            print(f"[IntelligentVoiceManager] ❌ Profile creation error: {e}")
            return False
    
    def _add_passive_sample(self, username, audio, confidence):
        """Add passive learning sample"""
        try:
            if username in known_users and confidence > 0.5:
                # Generate embedding and add to profile
                embedding = self._generate_current_embedding(audio)
                if embedding is not None:
                    self._add_embedding_to_profile(username, embedding)
                
                # Update recognition count
                profile = known_users[username]
                profile['recognition_count'] = profile.get('recognition_count', 0) + 1
                profile['last_updated'] = datetime.utcnow().isoformat()
                
                save_known_users()
                
                print(f"[IntelligentVoiceManager] 📈 Updated recognition count for {username}")
        except Exception as e:
            print(f"[IntelligentVoiceManager] ❌ Passive sample error: {e}")
    
    def update_current_speaker_name(self, name: str):
        """🔄 Update current speaker name after recognition"""
        try:
            if hasattr(self, 'current_speaker_cluster_id') and self.current_speaker_cluster_id:
                cluster_id = self.current_speaker_cluster_id
                
                # Update cluster with name
                from voice.database import link_anonymous_to_named
                if cluster_id in anonymous_clusters:
                    # Link to known user
                    success = link_anonymous_to_named(cluster_id, name)
                    if success:
                        print(f"[IntelligentVoiceManager] ✅ Linked {cluster_id} to {name}")
                        self.current_speaker_cluster_id = name
                
                print(f"[IntelligentVoiceManager] ✅ Updated speaker name: {name}")
                
        except Exception as e:
            print(f"[IntelligentVoiceManager] ❌ Error updating speaker name: {e}")

    def get_current_voice_embedding(self):
        """🔍 Get current voice embedding"""
        try:
            if hasattr(self, 'current_voice_embedding'):
                return self.current_voice_embedding
            return None
        except Exception as e:
            print(f"[IntelligentVoiceManager] ❌ Error getting voice embedding: {e}")
            return None
    
    def set_current_voice_embedding(self, audio):
        """🔄 Set current voice embedding from audio"""
        try:
            embedding = self._generate_current_embedding(audio)
            if embedding is not None:
                self.current_voice_embedding = embedding
                print(f"[IntelligentVoiceManager] 🔄 Updated current voice embedding")
            
        except Exception as e:
            print(f"[IntelligentVoiceManager] ❌ Error setting voice embedding: {e}")
    
    def set_current_cluster(self, cluster_id):
        """🔄 Set current speaker cluster"""
        try:
            self.current_speaker_cluster_id = cluster_id
            print(f"[IntelligentVoiceManager] 🔄 Set current cluster: {cluster_id}")
            
        except Exception as e:
            print(f"[IntelligentVoiceManager] ❌ Error setting cluster: {e}")
    
    def is_llm_locked(self):
        """Never lock LLM in intelligent mode"""
        return False
    
    def get_session_stats(self):
        """Get session statistics"""
        return {
            'interactions': self.interactions,
            'session_duration': (datetime.utcnow() - self.session_start).total_seconds(),
            'known_users': len(known_users),
            'anonymous_clusters': len(anonymous_clusters),
            'current_user': self.current_user,
            'system': 'IntelligentVoiceManager',
            'learning_history_profiles': len(self.voice_learning_history)
        }

# Global intelligent voice manager
voice_manager = IntelligentVoiceManager()