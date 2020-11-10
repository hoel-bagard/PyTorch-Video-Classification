# Script to crop and change the codec of all videos in the dataset. 
# Place this script in the dataset folder and run it

shopt -s globstar
# Crop and change file format / codec  (for dali)
for i in **/*.avi; do ffmpeg -i "$i" -filter:v "crop=2300:750:0:900" -pix_fmt yuv420p -b:v 4000k -c:v libx264 "${i%.*}.mp4"; done
# Remove the odl avi files
find . -name "*.avi" -type f -delete
