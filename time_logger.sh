#!/bin/bash
LOG_FILE="/home/kang/.ai-skills/time_log.txt"
echo "Time logging started at $(date)" > "$LOG_FILE"
while true; do
    echo "$(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
    sleep 180
done
