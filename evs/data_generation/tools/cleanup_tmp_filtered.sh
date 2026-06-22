# #!/usr/bin/env bash
# # cleanup_tmp_filtered.sh
# #
# # Finds all *__tmp_filtered.db3 directories, moves their contents into the
# # parent rosbag folder, then removes both the tmp dir and the original .db3 dir.
# #
# # Usage: ./cleanup_tmp_filtered.sh [root_dir]
# #   root_dir defaults to the current directory.
 
set -euo pipefail
 
ROOT="${1:-.}"
 
mapfile -t TMP_DIRS < <(find "$ROOT" -type d -name "*.db3")
 
if [[ ${#TMP_DIRS[@]} -eq 0 ]]; then
    echo "No *__tmp_filtered.db3 directories found under $ROOT"
    exit 0
fi
 
echo "Found ${#TMP_DIRS[@]} tmp dir(s) to clean up"
 
for tmp_dir in "${TMP_DIRS[@]}"; do
    rosbag_dir="$(dirname "$tmp_dir")"
    # Derive the original .db3 dir name by stripping __tmp_filtered
    original_name="${tmp_dir##*/}"          # basename
    # original_name="${original_name//__tmp_filtered/}"
    # original_dir="$rosbag_dir/$original_name"
 
    echo ""
    echo "  tmp:      $tmp_dir"
    # echo "  original: $original_dir"
    echo "  target:   $rosbag_dir"
 
    # Move contents of tmp dir into the rosbag folder
    mv "$tmp_dir"/* "$rosbag_dir"/
 
    # Remove both the (now empty) tmp dir and the original .db3 dir
    rm -rf "$tmp_dir"
    # rm -rf "$original_dir"
 
    echo "  done"
done
 
echo ""
echo "Cleanup complete."

# !/usr/bin/env bash
# cleanup_tmp_filtered.sh

# Finds all directories matching *__tmp_filtered.db3_0.db3, moves their
# contents into the parent rosbag folder renamed to <stem>.db3, then removes
# both the tmp dir and the original <stem>.db3 dir.

# Example:
#   Before:
#     rosbag/20240529_Takenaka_Murano_set3_0.db3          (original, removed)
#     rosbag/20240529_Takenaka_Murano_set3_0__tmp_filtered.db3_0.db3/  (tmp)
#   After:
#     rosbag/20240529_Takenaka_Murano_set3_0.db3/          (clean filtered bag)

# Usage: ./cleanup_tmp_filtered.sh [root_dir]

# set -euo pipefail

# ROOT="${1:-.}"

mapfile -t TMP_DIRS < <(find "$ROOT" -type f -name "*__tmp_filtered.db3_0.db3")

if [[ ${#TMP_DIRS[@]} -eq 0 ]]; then
    echo "No *__tmp_filtered.db3_0.db3 directories found under $ROOT"
    exit 0
fi

echo "Found ${#TMP_DIRS[@]} tmp dir(s) to clean up"

for tmp_dir in "${TMP_DIRS[@]}"; do
    rosbag_dir="$(dirname "$tmp_dir")"
    tmp_name="$(basename "$tmp_dir")"

    # Strip __tmp_filtered.db3_0.db3 → keep just <stem>.db3
    stem="${tmp_name//__tmp_filtered.db3_0.db3/}"
    clean_name="${stem}.db3"
    clean_dir="$rosbag_dir/$clean_name"
    original_dir="$rosbag_dir/$stem.db3"   # same as clean_dir, but explicit

    echo ""
    echo "  tmp:     $tmp_dir"
    echo "  target:  $clean_dir"

    # # Remove the original (unfiltered) dir if it exists
    # if [[ -d "$original_dir" && "$original_dir" != "$tmp_dir" ]]; then
    #     echo "  removing original: $original_dir"
    #     rm -rf "$original_dir"
    # fi

    # Rename tmp dir to the clean name
    mv "$tmp_dir" "$clean_dir"

    echo "  done → $clean_dir"
done

echo ""
echo "Cleanup complete."