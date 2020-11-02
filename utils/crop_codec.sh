# Script to crop and change the codec of all videos in the dataset. 
# Place this script in the dataset folder and run it (after uncommenting the following two lines)

# shopt -s globstar
# for i in **/*.avi; do ffmpeg -i "$i" -filter:v "crop=2300:750:0:900" -pix_fmt yuv420p -b:v 4000k -c:v libx264 "${i%.*}.mp4"; done
