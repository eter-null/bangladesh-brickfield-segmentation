from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import os

# Ask user for log directory path
log_dir = input("Enter the path to your TensorBoard log directory: ").strip()

# Remove quotes if user copied path with quotes
if log_dir.startswith('"') and log_dir.endswith('"'):
    log_dir = log_dir[1:-1]
elif log_dir.startswith("'") and log_dir.endswith("'"):
    log_dir = log_dir[1:-1]

# Check if directory exists
if not os.path.exists(log_dir):
    print(f"Error: Directory '{log_dir}' does not exist!")
    exit(1)

print(f"Loading TensorBoard data from: {log_dir}")

# Load the TensorBoard event data
try:
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
except Exception as e:
    print(f"Error loading TensorBoard data: {e}")
    exit(1)

# Extract scalar tags
scalar_tags = event_acc.Tags()['scalars']
print(f"Found {len(scalar_tags)} scalar tags:", scalar_tags)

if not scalar_tags:
    print("No scalar data found in the log directory!")
    exit(1)

# Ask for output directory (optional)
output_dir = input("Enter output directory name (default: 'scalar_exports'): ").strip()
if not output_dir:
    output_dir = "scalar_exports"

# Create output folder for CSVs
os.makedirs(output_dir, exist_ok=True)

# Export each scalar to its own CSV
print("Exporting scalars to CSV files...")
for i, tag in enumerate(scalar_tags, 1):
    try:
        events = event_acc.Scalars(tag)
        df = pd.DataFrame(events)
        # Clean the tag name for safe filename
        filename = tag.replace("/", "_").replace(":", "_").replace("\\", "_") + ".csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"[{i}/{len(scalar_tags)}] Exported: {filename}")
    except Exception as e:
        print(f"Error exporting {tag}: {e}")

print(f"\nSuccessfully exported {len(scalar_tags)} scalar files to '{output_dir}/'")