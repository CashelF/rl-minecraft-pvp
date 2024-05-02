#!/bin/bash

# Ensuring that MALMO_MINECRAFT_ROOT is set
if [ -z "$MALMO_MINECRAFT_ROOT" ]; then
    echo "The MALMO_MINECRAFT_ROOT environment variable is not set."
    exit 1
fi

# Starting the first Minecraft client on port 10000
echo "Launching the first Minecraft client on port 10000..."
"$MALMO_MINECRAFT_ROOT/launchClient.sh" -port 10000 &
first_pid=$!
echo "First Minecraft client launched with PID $first_pid"

# Waiting a bit to ensure the first client starts up properly
sleep 10

# Starting the second Minecraft client on port 10001
echo "Launching the second Minecraft client on port 10001..."
"$MALMO_MINECRAFT_ROOT/launchClient.sh" -port 10001 &
second_pid=$!
echo "Second Minecraft client launched with PID $second_pid"

# Waiting a bit to ensure the second client starts up properly
sleep 10

# Running the main Python script
echo "Running the Python script..."
python main.py

# wait $first_pid
# wait $second_pid

echo "Script execution completed."
