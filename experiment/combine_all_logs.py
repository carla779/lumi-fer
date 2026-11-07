#!/usr/bin/env python3
"""
Combine all FER frame-level logs (P002-P027) into a single CSV file.
Preserves all data rows without truncation.
"""
import os
import sys

# Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = current_dir
output_file = os.path.join(os.path.dirname(current_dir), "..", "..", "..", "all_fer_logs.csv")

# Ensure output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Participant IDs to process (P002 through P027, skip P001 and P030)
participants = [f"P{i:03d}" for i in range(2, 28)]

# Header for the combined file
output_header = "participant_id,timestamp,stimulus_id,fer_emotion,fer_confidence,conf_happy,conf_sad,conf_angry,conf_surprised,conf_neutral,face_detected"

total_frames = 0
participants_processed = []
participants_skipped = []

print("=" * 70)
print("FER Log Combination Script")
print("=" * 70)
print(f"Source directory: {source_dir}")
print(f"Output file: {output_file}")
print(f"Participants to process: {', '.join(participants)}")
print("=" * 70)
print()

# Write combined CSV
with open(output_file, 'w', newline='', encoding='utf-8') as outf:
    outf.write(output_header + '\n')

    for participant_id in participants:
        input_file = os.path.join(source_dir, f"{participant_id}_fer_log.csv")

        if not os.path.exists(input_file):
            print(f"⚠ {participant_id}: File not found, skipping")
            participants_skipped.append(participant_id)
            continue

        try:
            row_count = 0
            with open(input_file, 'r', encoding='utf-8') as inf:
                # Read all lines from file
                lines = inf.readlines()

                # Skip header (first line)
                if len(lines) > 1:
                    for line in lines[1:]:
                        line = line.rstrip('\n\r')
                        if line.strip():  # Skip empty lines
                            # Prepend participant_id to each data row
                            combined_line = f"{participant_id},{line}"
                            outf.write(combined_line + '\n')
                            total_frames += 1
                            row_count += 1

            participants_processed.append(f"{participant_id}: {row_count} frames")
            print(f"✓ {participant_id}: {row_count:6d} frames processed")

        except Exception as e:
            print(f"✗ {participant_id}: Error - {e}")
            participants_skipped.append(participant_id)

print()
print("=" * 70)
print("PROCESSING SUMMARY")
print("=" * 70)
print("\nSuccessfully processed:")
for p in participants_processed:
    print(f"  {p}")

if participants_skipped:
    print("\nSkipped:")
    for p in participants_skipped:
        print(f"  {p}")

print()
print("=" * 70)
print(f"Total frames combined: {total_frames:,}")
print(f"Output file: {output_file}")

# Check if file was created and get size
if os.path.exists(output_file):
    file_size = os.path.getsize(output_file)
    print(f"File size: {file_size / (1024*1024):.2f} MB ({file_size:,} bytes)")

    # Quick verification - count lines
    with open(output_file, 'r') as f:
        line_count = sum(1 for _ in f)
    print(f"Total lines in output: {line_count:,} (including header)")
else:
    print("ERROR: Output file was not created!")
    sys.exit(1)

print("=" * 70)
print("✓ Combination complete!")
