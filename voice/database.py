# voice/database.py - Voice profile database management with anonymous clustering
import json
import os
import time
import numpy as np  
import shutil       
from datetime import datetime
from config import KNOWN_USERS_PATH, DEBUG
# 🚀 ENHANCED: False positives tracking
false_positives = []

# ✅ WINDOWS COMPATIBILITY: Make fcntl optional
try:
    import fcntl
    FCNTL_AVAILABLE = True
except ImportError:
    FCNTL_AVAILABLE = False
    print("[Database]⚠️ fcntl not available on Windows - using alternative file locking")


# Global voice database
known_users = {}
anonymous_clusters = {}  # ✅ NEW: Anonymous voice clusters
cluster_counter = 1

def load_known_users():
    """🚀 ENHANCED: Load known users and anonymous clusters with false positives tracking"""
    global known_users, anonymous_clusters, false_positives
    
    try:
        print(f"[DEBUG] 📂 LOAD_KNOWN_USERS called at {datetime.utcnow().isoformat()}")
        print(f"[DEBUG] 📂 Loading from: {KNOWN_USERS_PATH}")
        print(f"[DEBUG] 📂 File exists: {os.path.exists(KNOWN_USERS_PATH)}")
        
        if os.path.exists(KNOWN_USERS_PATH):
            file_size = os.path.getsize(KNOWN_USERS_PATH)
            print(f"[DEBUG] 📊 File size: {file_size} bytes")
            
            with open(KNOWN_USERS_PATH, 'r') as f:
                data = json.load(f)
            
            known_users = data.get('known_users', {})
            anonymous_clusters = data.get('anonymous_clusters', {})
            false_positives = data.get('false_positives', [])  # 🚀 NEW!
            
            print(f"[DEBUG] ✅ LOAD SUCCESSFUL:")
            print(f"[DEBUG] 👥 Known users: {len(known_users)} - {list(known_users.keys())}")
            print(f"[DEBUG] 🔍 Anonymous clusters: {len(anonymous_clusters)} - {list(anonymous_clusters.keys())}")
            print(f"[DEBUG] 🚨 False positives: {len(false_positives)} entries")
            
            # Show cluster details
            for cluster_id, cluster_data in anonymous_clusters.items():
                print(f"[DEBUG] 🔍 Cluster {cluster_id}: {cluster_data.get('sample_count', 0)} samples")
            
            # Show false positive summary
            if false_positives:
                recent_fps = false_positives[-5:]  # Show last 5
                print(f"[DEBUG] 🚨 Recent false positives: {[fp.get('blocked_name', 'unknown') for fp in recent_fps]}")
            
            print(f"[DEBUG] 📅 Last updated: {data.get('last_updated', 'unknown')}")
            return True
        else:
            print(f"[DEBUG] ⚠️ File does not exist - initializing empty dictionaries")
            known_users = {}
            anonymous_clusters = {}
            false_positives = []
            return False
            
    except Exception as e:
        print(f"[DEBUG] ❌ LOAD_KNOWN_USERS ERROR: {e}")
        import traceback
        traceback.print_exc()
        known_users = {}
        anonymous_clusters = {}
        false_positives = []
        return False

def convert_numpy_for_json(obj):
    """Convert numpy types to JSON-serializable types recursively"""
    import numpy as np
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_, np.bool8)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_for_json(item) for item in obj]
    else:
        return obj

def save_known_users():
    """🚀 ENHANCED: Save known users and anonymous clusters with false positives tracking"""
    try:
        debug_database_state()  # 🔥 ADD DEBUG CALL HERE
        
        print(f"[DEBUG] 💾 SAVE_KNOWN_USERS CALLED at {datetime.utcnow().isoformat()}")
        print(f"[DEBUG] 👥 Known users count: {len(known_users)}")
        print(f"[DEBUG] 🔍 Anonymous clusters count: {len(anonymous_clusters)}")
        print(f"[DEBUG] 🚨 False positives count: {len(false_positives)}")
        print(f"[DEBUG] 📊 Known users keys: {list(known_users.keys())}")
        print(f"[DEBUG] 📊 Cluster keys: {list(anonymous_clusters.keys())}")
        
        # Show cluster details
        for cluster_id, cluster_data in anonymous_clusters.items():
            print(f"[DEBUG] 🔍 Cluster {cluster_id}: {cluster_data.get('sample_count', 0)} samples, created {cluster_data.get('created_at', 'unknown')}")
        
        # 🔧 ENHANCED: Convert numpy types before JSON serialization
        print(f"[DEBUG] 🔄 Converting numpy types for JSON serialization...")
        
        try:
            clean_known_users = convert_numpy_for_json(known_users)
            print(f"[DEBUG] ✅ Known users conversion successful")
        except Exception as e:
            print(f"[DEBUG] ❌ Known users conversion failed: {e}")
            clean_known_users = {}
        
        try:
            clean_anonymous_clusters = convert_numpy_for_json(anonymous_clusters)
            print(f"[DEBUG] ✅ Anonymous clusters conversion successful")
        except Exception as e:
            print(f"[DEBUG] ❌ Anonymous clusters conversion failed: {e}")
            clean_anonymous_clusters = {}
        
        try:
            clean_false_positives = convert_numpy_for_json(false_positives)
            print(f"[DEBUG] ✅ False positives conversion successful")
        except Exception as e:
            print(f"[DEBUG] ❌ False positives conversion failed: {e}")
            clean_false_positives = []
        
        # 🚀 ENHANCED: Create data structure with false positives
        data = {
            'known_users': clean_known_users,
            'anonymous_clusters': clean_anonymous_clusters,
            'false_positives': clean_false_positives,  # 🚀 NEW!
            'last_updated': datetime.utcnow().isoformat(),
            'version': '2.1_enhanced_protection'
        }
        
        print(f"[DEBUG] 📝 Writing to: {KNOWN_USERS_PATH}")
        print(f"[DEBUG] 📝 Data structure: known_users={len(data['known_users'])}, clusters={len(data['anonymous_clusters'])}, false_positives={len(data['false_positives'])}")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(KNOWN_USERS_PATH), exist_ok=True)
        
        # 🔧 ENHANCED: Test JSON serialization before writing
        try:
            test_json = json.dumps(data, indent=2, ensure_ascii=False)
            print(f"[DEBUG] ✅ JSON serialization test passed - {len(test_json)} characters")
        except Exception as json_error:
            print(f"[DEBUG] ❌ JSON serialization test failed: {json_error}")
            
            # Try to identify the problematic data
            for key, value in data.items():
                try:
                    json.dumps({key: value})
                    print(f"[DEBUG] ✅ {key} is JSON serializable")
                except Exception as key_error:
                    print(f"[DEBUG] ❌ {key} is NOT JSON serializable: {key_error}")
            
            raise json_error
        
        # Write the file
        with open(KNOWN_USERS_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        debug_database_state()  # 🔥 ADD DEBUG CALL HERE
        
        print(f"[DEBUG] ✅ SAVE SUCCESSFUL - File written with {len(data['known_users'])} users, {len(data['anonymous_clusters'])} clusters, {len(data['false_positives'])} false positives")
        
        # Verify file was written
        if os.path.exists(KNOWN_USERS_PATH):
            file_size = os.path.getsize(KNOWN_USERS_PATH)
            print(f"[DEBUG] 📊 File size: {file_size} bytes")
            
            # Read back and verify
            try:
                with open(KNOWN_USERS_PATH, 'r', encoding='utf-8') as f:
                    verify_data = json.load(f)
                print(f"[DEBUG] ✅ Verification: {len(verify_data.get('known_users', {}))} users, {len(verify_data.get('anonymous_clusters', {}))} clusters, {len(verify_data.get('false_positives', []))} false positives")
                print(f"[DEBUG] ✅ Version: {verify_data.get('version', 'unknown')}")
            except Exception as verify_error:
                print(f"[DEBUG] ❌ Verification failed: {verify_error}")
        else:
            print(f"[DEBUG] ❌ File does not exist after write attempt!")
            
        return True
        
    except Exception as e:
        print(f"[DEBUG] ❌ SAVE_KNOWN_USERS ERROR: {e}")
        print(f"[DEBUG] ❌ Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        
        # 🔧 EMERGENCY FALLBACK: Try to save without problematic data
        try:
            print(f"[DEBUG] 🚨 Attempting emergency fallback save...")
            
            fallback_data = {
                'known_users': {},
                'anonymous_clusters': {},
                'false_positives': [],
                'last_updated': datetime.utcnow().isoformat(),
                'version': '2.1_enhanced_protection_fallback',
                'original_error': str(e)
            }
            
            with open(KNOWN_USERS_PATH, 'w', encoding='utf-8') as f:
                json.dump(fallback_data, f, indent=2, ensure_ascii=False)
            
            print(f"[DEBUG] ✅ Emergency fallback save successful")
            return False  # Indicate partial failure
            
        except Exception as fallback_error:
            print(f"[DEBUG] ❌ Emergency fallback also failed: {fallback_error}")
            return False

def _convert_numpy_to_list(data):
    """Convert numpy arrays to lists recursively with proper error handling"""
    try:
        # Handle numpy arrays
        if isinstance(data, np.ndarray):
            return data.tolist()
        
        # Handle dictionaries recursively
        elif isinstance(data, dict):
            return {key: _convert_numpy_to_list(value) for key, value in data.items()}
        
        # Handle lists recursively
        elif isinstance(data, list):
            return [_convert_numpy_to_list(item) for item in data]
        
        # Handle other numpy types (scalars, etc.)
        elif hasattr(data, 'tolist') and callable(getattr(data, 'tolist')):
            return data.tolist()
        
        # Return as-is for JSON-serializable types
        else:
            return data
            
    except Exception as e:
        print(f"[Database] ⚠️ Error converting data to list: {e}")
        print(f"[Database] ⚠️ Data type: {type(data)}")
        # Fallback: try to convert to basic Python types
        if hasattr(data, 'tolist'):
            try:
                return data.tolist()
            except:
                pass
        return str(data)  # Last resort: convert to string

def create_anonymous_cluster(embedding, quality_info=None):
    """🆕 Create anonymous cluster with sequential naming: Anonymous_01, Anonymous_02, etc."""
    try:
        print(f"[DEBUG] 🆕 CREATE_ANONYMOUS_CLUSTER called at {datetime.utcnow().isoformat()}")
        
        # ✅ FIXED: Find highest existing anonymous number
        max_num = 0
        for cluster_id in anonymous_clusters.keys():
            if cluster_id.startswith('Anonymous_'):
                try:
                    # Extract number from Anonymous_001, Anonymous_002, etc.
                    num_str = cluster_id.split('_')[1]
                    num = int(num_str)
                    max_num = max(max_num, num)
                except (IndexError, ValueError):
                    continue
        
        # Create next sequential ID with zero-padding
        next_num = max_num + 1
        cluster_id = f"Anonymous_{next_num:03d}"  # Anonymous_001, Anonymous_002, etc.
        
        print(f"[DEBUG] 🏷️ Generated sequential cluster ID: {cluster_id}")
        
        # Validate embedding
        if embedding is None:
            print(f"[DEBUG] ❌ Embedding is None - cannot create cluster")
            return None
        
        # Prepare cluster data
        cluster_data = {
            'cluster_id': cluster_id,
            'embeddings': [embedding],
            'created_at': datetime.utcnow().isoformat(),
            'last_updated': datetime.utcnow().isoformat(),
            'sample_count': 1,
            'status': 'anonymous',
            'quality_scores': [quality_info.get('overall_score', 0.5)] if quality_info else [0.5],
            'audio_contexts': ['unknown_speaker'],
            'confidence_threshold': 0.6,
            'clustering_metrics': [quality_info] if quality_info else [{'clustering_suitability': 'unknown'}]
        }
        
        print(f"[DEBUG] 📝 Cluster data prepared with {len(cluster_data['embeddings'])} embeddings")
        
        # Add to global dictionary
        anonymous_clusters[cluster_id] = cluster_data
        print(f"[DEBUG] ✅ Cluster added to global dictionary")
        print(f"[DEBUG] 📊 New cluster count: {len(anonymous_clusters)}")
        print(f"[DEBUG] 📊 All clusters: {list(anonymous_clusters.keys())}")
        
        # ✅ CRITICAL: Save immediately after creation
        print(f"[DEBUG] 💾 Calling save_known_users() for cluster: {cluster_id}")
        save_result = save_known_users()
        print(f"[DEBUG] 💾 Save result: {save_result}")
        
        if save_result:
            print(f"[DEBUG] ✅ SEQUENTIAL CLUSTER CREATION SUCCESSFUL: {cluster_id}")
        else:
            print(f"[DEBUG] ❌ CLUSTER SAVE FAILED: {cluster_id}")
        
        return cluster_id
        
    except Exception as e:
        print(f"[DEBUG] ❌ CREATE_ANONYMOUS_CLUSTER ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def link_anonymous_to_named(cluster_id, username):
    """🔗 Enhanced link with proper cluster data transfer"""
    global known_users, anonymous_clusters
    
    try:
        print(f"[DEBUG] 🔗 LINK_ANONYMOUS_TO_NAMED called: {cluster_id} → {username}")
        
        if cluster_id not in anonymous_clusters:
            print(f"[DEBUG] ❌ Cluster {cluster_id} not found in anonymous_clusters")
            print(f"[DEBUG] 📊 Available clusters: {list(anonymous_clusters.keys())}")
            return False
        
        cluster_data = anonymous_clusters[cluster_id]
        print(f"[DEBUG] 📊 Cluster data: {cluster_data.get('sample_count', 0)} samples")
        
        # ✅ ENHANCED: Handle name collision
        final_username = handle_same_name_collision(username)
        if final_username != username:
            print(f"[DEBUG] 🔄 Name collision resolved: {username} → {final_username}")
        
        # ✅ GET ALL EMBEDDINGS from anonymous cluster
        all_embeddings = cluster_data.get('embeddings', [])
        
        if not all_embeddings:
            print(f"[DEBUG] ❌ No embeddings found in cluster {cluster_id}")
            return False
        
        print(f"[DEBUG] 📊 Transferring {len(all_embeddings)} embeddings from {cluster_id} to {final_username}")
        
        # Check if user already exists
        if final_username in known_users:
            print(f"[DEBUG] 🔗 Merging with existing user: {final_username}")
            
            # Merge with existing user
            existing_embeddings = known_users[final_username].get('embeddings', [])
            
            # Combine embeddings (max 15 total)
            combined_embeddings = existing_embeddings + all_embeddings
            if len(combined_embeddings) > 15:
                combined_embeddings = combined_embeddings[-15:]  # Keep most recent
            
            # Update existing user
            known_users[final_username]['embeddings'] = combined_embeddings
            known_users[final_username]['last_updated'] = datetime.utcnow().isoformat()
            known_users[final_username]['anonymous_samples_merged'] = len(all_embeddings)
            known_users[final_username]['total_samples'] = len(combined_embeddings)
            
            print(f"[DEBUG] 🔗 Merged {cluster_id} into existing {final_username}")
        else:
            print(f"[DEBUG] 🆕 Creating new user: {final_username}")
            
            # Create new named user from cluster with ALL data
            known_users[final_username] = {
                'username': final_username,
                'name': final_username,
                'embeddings': all_embeddings,  # ✅ TRANSFER ALL EMBEDDINGS
                'created_at': cluster_data.get('created_at'),
                'last_updated': datetime.utcnow().isoformat(),
                'status': 'trained',
                'confidence_threshold': 0.4,  # Lower threshold for identified users
                'quality_scores': cluster_data.get('quality_scores', []),
                'sample_count': cluster_data.get('sample_count', len(all_embeddings)),
                'original_cluster': cluster_id,
                'recognition_count': 0,
                'background_learning': True,
                'voice_profile_complete': True,
                'linked_from_anonymous': True,
                'embedding_version': '2.0_anonymous_transfer'
            }
            
            print(f"[DEBUG] 🎯 Created {final_username} from {cluster_id} with {len(all_embeddings)} embeddings")
        
        # ✅ REMOVE anonymous cluster safely
        try:
            del anonymous_clusters[cluster_id]
            print(f"[DEBUG] 🗑️ Removed anonymous cluster: {cluster_id}")
        except KeyError:
            print(f"[DEBUG] ⚠️ Cluster {cluster_id} already removed")
        
        # ✅ SAVE with verification
        save_result = save_known_users()
        if save_result:
            print(f"[DEBUG] ✅ LINK SUCCESSFUL: {cluster_id} → {final_username}")
            return True
        else:
            print(f"[DEBUG] ❌ LINK SAVE FAILED: {cluster_id} → {final_username}")
            return False
            
    except Exception as e:
        print(f"[DEBUG] ❌ LINK_ANONYMOUS_TO_NAMED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def add_false_positive(entry):
    """🚀 NEW: Add false positive entry to tracking"""
    global false_positives
    
    try:
        # Add timestamp if not present
        if 'timestamp' not in entry:
            entry['timestamp'] = datetime.utcnow().isoformat()
        
        # Add to false positives list
        false_positives.append(entry)
        
        # Keep only last 1000 entries (prevent memory bloat)
        if len(false_positives) > 1000:
            false_positives = false_positives[-1000:]
        
        print(f"[DEBUG] 🚨 False positive added: {entry.get('blocked_name', 'unknown')}")
        print(f"[DEBUG] 📊 Total false positives: {len(false_positives)}")
        
        return True
        
    except Exception as e:
        print(f"[DEBUG] ❌ ADD_FALSE_POSITIVE ERROR: {e}")
        return False

def get_false_positive_stats():
    """🚀 NEW: Get false positive statistics"""
    global false_positives
    
    try:
        if not false_positives:
            return {
                'total_count': 0,
                'recent_count': 0,
                'most_common_blocked': [],
                'most_common_reasons': []
            }
        
        # Recent false positives (last 24 hours)
        recent_cutoff = datetime.utcnow() - timedelta(hours=24)
        recent_fps = [
            fp for fp in false_positives 
            if datetime.fromisoformat(fp.get('timestamp', '1970-01-01T00:00:00')) > recent_cutoff
        ]
        
        # Most common blocked names
        from collections import Counter
        blocked_names = [fp.get('blocked_name', 'unknown') for fp in false_positives]
        most_common_blocked = Counter(blocked_names).most_common(5)
        
        # Most common reasons
        all_reasons = []
        for fp in false_positives:
            reasons = fp.get('suspicious_reasons', [])
            if isinstance(reasons, list):
                all_reasons.extend(reasons)
        
        most_common_reasons = Counter(all_reasons).most_common(5)
        
        return {
            'total_count': len(false_positives),
            'recent_count': len(recent_fps),
            'most_common_blocked': most_common_blocked,
            'most_common_reasons': most_common_reasons,
            'last_updated': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        print(f"[DEBUG] ❌ GET_FALSE_POSITIVE_STATS ERROR: {e}")
        return {'error': str(e)}

def protect_existing_cluster_name(cluster_id, new_name):
    """🚀 NEW: Protect existing cluster names from overwrite"""
    global known_users, anonymous_clusters
    
    try:
        # Check if cluster exists and has a name
        if cluster_id in known_users:
            existing_name = known_users[cluster_id].get('name')
            if existing_name and existing_name != new_name:
                print(f"[DEBUG] 🛡️ CLUSTER NAME PROTECTION: {cluster_id} has existing name '{existing_name}', blocking overwrite to '{new_name}'")
                return False, existing_name
        
        # Allow name assignment
        return True, None
        
    except Exception as e:
        print(f"[DEBUG] ❌ PROTECT_EXISTING_CLUSTER_NAME ERROR: {e}")
        return False, None

def cleanup_false_positives(max_age_days=30):
    """🚀 NEW: Clean up old false positive entries"""
    global false_positives
    
    try:
        if not false_positives:
            return 0
        
        cutoff_time = datetime.utcnow() - timedelta(days=max_age_days)
        
        original_count = len(false_positives)
        false_positives = [
            fp for fp in false_positives
            if datetime.fromisoformat(fp.get('timestamp', '1970-01-01T00:00:00')) > cutoff_time
        ]
        
        cleaned_count = original_count - len(false_positives)
        
        if cleaned_count > 0:
            print(f"[DEBUG] 🧹 Cleaned up {cleaned_count} old false positives")
            save_known_users()
        
        return cleaned_count
        
    except Exception as e:
        print(f"[DEBUG] ❌ CLEANUP_FALSE_POSITIVES ERROR: {e}")
        return 0

def find_similar_clusters(embedding, threshold=0.5):
    """✅ NEW: Find similar anonymous clusters for merging"""
    similar_clusters = []
    
    try:
        from voice.voice_models import dual_voice_model_manager
        
        for cluster_id, cluster_data in anonymous_clusters.items():
            cluster_embeddings = cluster_data.get('embeddings', [])
            
            for stored_embedding in cluster_embeddings:
                similarity = dual_voice_model_manager.compare_dual_embeddings(
                    embedding, stored_embedding
                )
                
                if similarity > threshold:
                    similar_clusters.append({
                        'cluster_id': cluster_id,
                        'similarity': similarity,
                        'embedding_count': len(cluster_embeddings)
                    })
                    break
    except:
        pass
    
    return sorted(similar_clusters, key=lambda x: x['similarity'], reverse=True)

def get_voice_display_name(user_id):
    """Get display name for voice-identified user"""
    try:
        if user_id in known_users:
            profile = known_users[user_id]
            if isinstance(profile, dict):
                return profile.get('display_name', profile.get('real_name', user_id))
        return user_id
    except Exception as e:
        print(f"[Database] ⚠️ Error getting display name: {e}")
        return user_id

def update_voice_profile_display_name(user_id, display_name):
    """Update display name for voice profile"""
    try:
        if user_id in known_users:
            if isinstance(known_users[user_id], dict):
                known_users[user_id]['display_name'] = display_name
            else:
                # Convert old format to new
                old_embedding = known_users[user_id]
                known_users[user_id] = {
                    'embedding': old_embedding,
                    'display_name': display_name,
                    'created_date': time.time()
                }
            save_known_users()
            return True
    except Exception as e:
        print(f"[Database] ❌ Error updating display name: {e}")
    return False

def handle_same_name_collision(username):
    """✅ NEW: Handle multiple users with same name"""
    if username not in known_users:
        return username
    
    # Find next available suffix
    counter = 2
    while f"{username}_{counter:03d}" in known_users:
        counter += 1
    
    new_username = f"{username}_{counter:03d}"
    print(f"[Database] 🔄 Same name collision: {username} → {new_username}")
    return new_username

def get_all_clusters():
    """✅ NEW: Get all voice clusters (named + anonymous)"""
    all_clusters = {}
    all_clusters.update(known_users)
    all_clusters.update(anonymous_clusters)
    return all_clusters

def cleanup_old_anonymous_clusters(max_age_days=7):
    """🧹 Clean up old anonymous clusters"""
    try:
        from datetime import datetime, timedelta
        
        current_time = datetime.utcnow()
        cutoff_time = current_time - timedelta(days=max_age_days)
        
        clusters_to_remove = []
        for cluster_id, cluster_data in anonymous_clusters.items():
            try:
                created_at = datetime.fromisoformat(cluster_data.get('created_at', current_time.isoformat()))
                if created_at < cutoff_time:
                    clusters_to_remove.append(cluster_id)
            except:
                # Remove clusters with invalid timestamps
                clusters_to_remove.append(cluster_id)
        
        for cluster_id in clusters_to_remove:
            del anonymous_clusters[cluster_id]
            print(f"[Database] 🧹 Cleaned up old cluster: {cluster_id}")
        
        if clusters_to_remove:
            save_known_users()
        
        return len(clusters_to_remove)
        
    except Exception as e:
        print(f"[Database] ❌ Cleanup error: {e}")
        return 0

def debug_database_state():
    """🐛 Debug current database state"""
    print(f"\n🐛 DATABASE DEBUG:")
    print(f"📊 known_users: {list(known_users.keys())}")
    print(f"📊 anonymous_clusters: {list(anonymous_clusters.keys())}")
    print(f"📊 known_users id: {id(known_users)}")
    print(f"📊 anonymous_clusters id: {id(anonymous_clusters)}")
    
    # Check if file exists and what it contains
    try:
        with open(KNOWN_USERS_PATH, 'r') as f:
            data = json.load(f)
        print(f"📁 File has: {len(data.get('known_users', {}))} users, {len(data.get('anonymous_clusters', {}))} clusters")
    except:
        print(f"📁 File doesn't exist or is corrupted")
    print()

# ✅ Load database on import
load_known_users()