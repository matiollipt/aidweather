# AidWeather Installation Tests - Docker Container Ubuntu Latest

## 1. Launch the Ubuntu container
docker run -it --name aidweather-env ubuntu:latest

## --- RUN THE FOLLOWING COMMANDS INSIDE THE CONTAINER TERMINAL ---

## 2. Update and install download utilities
apt-get update
apt-get install -y curl git python3 python3-venv python3-pip 

## 3. Download and execute the installer script (Update URL to matches vendor source)
curl -fsSL https://raw.githubusercontent.com/matiollipt/aidweather/main/install.sh | bash -s -- --dev -y

# 4. Run verification tests
cd aidweather
source .venv/bin/activate
python3 -c "import aidweather; print(aidweather.__version__)"

