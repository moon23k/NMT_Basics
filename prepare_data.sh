#!/bin/bash
mkdir -p data
cd data
mkdir -p seq ids vocab


#Download Data
echo "Data Download Started"
python3 ../data_processing/download_data.py
echo "Data Download Completed"


#Get sentencepiece
echo "Downloading Sentencepiece"
git clone https://github.com/google/sentencepiece.git
cd sentencepiece
mkdir build
cd build
cmake ..
make -j $(nproc)
sudo make install
sudo ldconfig
cd ../../


#Build Sentencepice Vocab and Model
echo "Building Vocab Started"
cat seq/* > concat.txt
bash ../data_processing/build_vocab.sh -i concat.txt -p vocab/spm
rm concat.txt
echo "Building Vocab Completed"


#Tokens to Ids
splits=(train valid test)
extensions=(src trg)
for split in "${splits[@]}"; do
    for ext in "${extensions[@]}"; do
        spm_encode --model=vocab/spm.model --extra_options=bos:eos \
        --output_format=id < seq/${split}.${ext} > ids/${split}.${ext}
        echo " Converting Tokens to Ids on ${split}.${ext} has completed"
    done
done


rm -rf sentencepiece