#/bin/bsh

find  $1 -name '*.jpg' >> images.txt
cat images.txt | grep autofocus_0 >> images_autofocus.txt
cat images.txt | grep autofocus_need >> images_autofocus_needle.txt
cat images.txt | grep -E 'conc_.{7}\/' >> images_data.txt
