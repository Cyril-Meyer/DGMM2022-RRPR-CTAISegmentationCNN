FILES="./DGMM2022-MEYER-DATA/I3_IMAGE_172-251_LOWPASS/*"
for f in $FILES
do
  if [ -f "$f" ]
  then
    echo "Processing $f file..."
    ./02_preprocess_ComponentTreeAttributeImage $f
  fi
done

