# Setup

1. Install the [Minecraft Launcher](https://www.minecraft.net/en-us/download)

2. Install the dependencies for MalmO, found [here](https://github.com/Microsoft/malmo/blob/master/scripts/python-wheel/README.md)


3. Install the necessary dependencies: 
```bash
conda env create -f environment.yaml
```

4. Activate the environment: 
```bash
conda activate rl
```

5. Launch two minecraft instances and begin learning:

On Windows:
```bash
start launchClient.bat -port 10000
start launchClient.bat -port 10001
```
On Linux:
```bash
$MALMO_MINECRAFT_ROOT/launchClient.sh -port 10000
$MALMO_MINECRAFT_ROOT/launchClient.sh -port 10001
```

Begin learning:
```bash
python main.py
```

Alternatively, instead of 5,
```bash
chmod +x run.sh
./run.sh
```