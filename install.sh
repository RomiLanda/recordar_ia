sudo apt-get -y install tesseract-ocr==4;
wget https://github.com/tesseract-ocr/tessdata_best/raw/main/spa.traineddata;
sudo mv spa.traineddata /usr/share/tesseract-ocr/4.00/tessdata/;

mkdir input_data;
mkdir out_data;
sudo chmod 777 input_data;
sudo chmod 777 out_data;