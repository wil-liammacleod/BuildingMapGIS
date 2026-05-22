import hashlib
import json
from pathlib import Path
import os

def get_file_hash(path: Path) -> str:
    """Calculates the SHA-256 hash of a file on disk."""
    if not path.exists():
        return ""
    
    sha256 = hashlib.sha256()
    # Read in chunks of 64KB
    with open(path, 'rb') as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()

def get_config_hash(config_dict: dict) -> str:
    """Helper to hash configuration dictionaries stably."""
    # Serialize to JSON with sorted keys to ensure stable formatting
    serialized = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()

def should_run_stage(
    stage_name: str,
    input_files: list,
    config: dict,
    output_files: list,
    cache_file: Path,
    force: bool = False
) -> bool:
    """
    Checks if a pipeline stage needs to be executed.
    Returns True if the stage should run, and False if it can be skipped.
    """
    if force:
        print(f"🔄 Stage '{stage_name}': Forced execution requested.")
        return True

    # Ensure all output files actually exist
    for out_file in output_files:
        out_path = Path(out_file)
        if not out_path.exists():
            print(f"🔄 Stage '{stage_name}': Output file missing ({out_path.name}). Running stage.")
            return True

    # If cache file doesn't exist, we must run
    if not cache_file.exists():
        print(f"🔄 Stage '{stage_name}': No cache file found. Running stage.")
        return True

    # Load cache metadata
    try:
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
    except Exception as e:
        print(f"⚠️ Error reading cache file: {e}. Re-running stage.")
        return True

    stages_cache = cache_data.get("stages", {})
    if stage_name not in stages_cache:
        print(f"🔄 Stage '{stage_name}': No cached metadata found. Running stage.")
        return True

    stage_entry = stages_cache[stage_name]

    # Verify configuration hash
    current_config_hash = get_config_hash(config)
    cached_config_hash = stage_entry.get("config_hash", "")
    if current_config_hash != cached_config_hash:
        print(f"🔄 Stage '{stage_name}': Configuration changed. Running stage.")
        return True

    # Verify input file hashes
    cached_input_hashes = stage_entry.get("input_hashes", {})
    for inp in input_files:
        inp_path = Path(inp)
        if not inp_path.exists():
            print(f"🔄 Stage '{stage_name}': Input file missing ({inp_path.name}). Running stage.")
            return True
        
        current_hash = get_file_hash(inp_path)
        cached_hash = cached_input_hashes.get(str(inp_path.resolve()), "")
        
        if current_hash != cached_hash:
            print(f"🔄 Stage '{stage_name}': Input file changed ({inp_path.name}). Running stage.")
            return True

    print(f"⏭️  Stage '{stage_name}': All inputs and configuration match cache. Skipping.")
    return False

def update_stage_cache(
    stage_name: str,
    input_files: list,
    config: dict,
    cache_file: Path
) -> None:
    """Updates the cache file metadata upon successful stage completion."""
    cache_data = {"stages": {}}
    
    # Load existing cache data if available
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
        except Exception:
            pass
            
    if "stages" not in cache_data:
        cache_data["stages"] = {}

    # Compute hashes
    config_hash = get_config_hash(config)
    input_hashes = {}
    for inp in input_files:
        inp_path = Path(inp)
        if inp_path.exists():
            input_hashes[str(inp_path.resolve())] = get_file_hash(inp_path)

    # Save to stages cache
    cache_data["stages"][stage_name] = {
        "config_hash": config_hash,
        "input_hashes": input_hashes
    }

    # Write back to cache file atomically
    temp_file = cache_file.with_suffix('.tmp')
    try:
        # Create directories if they don't exist
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        os.replace(temp_file, cache_file)
    except Exception as e:
        print(f"⚠️ Failed to write cache metadata file: {e}")
        if temp_file.exists():
            try:
                temp_file.unlink()
            except Exception:
                pass
