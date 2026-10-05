echo "- Downloading ImageNet 2012 dataset (Imagenet 1k)"
mkdir -p ~/data
cd ~/data
if [[ ! -d 'imagenet' ]]; then
    mkdir -p imagenet
    cd imagenet
    echo "- Downloading ILSVRC2012_img_train.tar"
    wget --no-check-certificate  https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
    echo "- Downloading ILSVRC2012_img_val.tar"
    wget --no-check-certificate  https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
    echo "- Extracting validation set"
    mkdir -p val
    tar -xvf ILSVRC2012_img_val.tar -C val
    cd val
    wget https://raw.githubusercontent.com/PatrickHua/EasyImageNet/main/valprep.sh
    sh valprep.sh
    cd ..
    echo "- Extracting training set"
    mkdir -p train
    tar -xvf ILSVRC2012_img_train.tar -C train
    cd train
    find . -name "*.tar" | while read NAME ; do
        mkdir -p "${NAME%.tar}"
        tar -xvf "${NAME}" -C "${NAME%.tar}"
        rm -f "${NAME}"
    done
    cd ..
    cd ..
fi