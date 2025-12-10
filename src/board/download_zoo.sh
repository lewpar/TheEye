# Setup Zoo Path
mkdir ./zoo

# Get HAILO Inferencing Models
echo "Downloading HAILO models.."
mkdir ./zoo/hailo
degirum download-zoo --url "https://hub.degirum.com/degirum/hailo" --path "./zoo/hailo"
echo "Finished downloading HAILO models."