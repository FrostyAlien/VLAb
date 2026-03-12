import os
from pathlib import Path

# Paths to the specific snapshots we found
base_v1 = Path("/home/alien/.cache/huggingface/hub/datasets--HuggingFaceVLA--community_dataset_v1/snapshots/9aec91fdb060ef116fb200abb17e3c0994147c80")
base_v2 = Path("/home/alien/.cache/huggingface/hub/datasets--HuggingFaceVLA--community_dataset_v2/snapshots/815dd11af21a488d2c8c6a87e550d6e890fbf558")

lerobot_cache = Path("/home/alien/.cache/huggingface/lerobot")
lerobot_cache.mkdir(parents=True, exist_ok=True)

repo_ids = []

def process_base(base_path):
    print(f"Processing {base_path}...")
    # Find all info.json files
    for root, dirs, files in os.walk(base_path):
        if "info.json" in files and os.path.basename(root) == "meta":
            # root is .../user/dataset/meta
            dataset_dir = Path(root).parent
            # Relative path from base gives user/dataset
            rel_path = dataset_dir.relative_to(base_path)
            repo_id = str(rel_path)
            
            repo_ids.append(repo_id)
            
            # Target link location
            link_target = lerobot_cache / rel_path
            
            # Create parent dir (user dir)
            link_target.parent.mkdir(parents=True, exist_ok=True)
            
            # Remove existing link/dir if it exists
            if link_target.exists():
                if link_target.is_symlink():
                    link_target.unlink()
                elif link_target.is_dir():
                    # Check if empty, if so remove? Or warn?
                    # For safety, let's just warn and skip if it's a real dir
                    print(f"Warning: {link_target} is a real directory, skipping symlink.")
                    continue
            
            # Create symlink: link_target -> dataset_dir
            print(f"Linking {repo_id}")
            try:
                os.symlink(dataset_dir, link_target)
            except OSError as e:
                print(f"Failed to link {repo_id}: {e}")

process_base(base_v1)
process_base(base_v2)

print("\n" + "="*80)
print("ALL OK! Sub-datasets linked.")
print("="*80)

# Print the full repo_id string
full_repo_id_str = ",".join(repo_ids)
# print(f"\nFULL REPO_ID STRING:\n{full_repo_id_str}")

# Save to a file for easy reading
with open("repo_ids.txt", "w") as f:
    f.write(full_repo_id_str)
print("Repo IDs saved to repo_ids.txt")
