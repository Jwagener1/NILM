# Setup the beaglebone AI
sudo apt update -y
sudo apt upgrade -y
sudo reboot

sudo /opt/scripts/tools/update_kernel.sh --lts-4_19
sudo reboot

sudo /opt/scripts/tools/developers/update_bootloader.sh
sudo reboot

cd
git clone https://github.com/beagleboard/BeagleBoard-DeviceTrees
cd BeagleBoard-DeviceTrees
make

sudo cp src/arm/am5729-beagleboneai.dtb /boot/dtbs/4.19.94-ti-r73/

sudo cp src/arm/overlays/BBAI-PRUOUT_PRU1_0.dtbo /lib/firmware/

sudo cp src/arm/overlays/BBAI-PRUOUT_PRU1_1.dtbo /lib/firmware/

sudo cp src/arm/overlays/BBAI-PRUIN_PRU1_0.dtbo /lib/firmware/

sudo cp src/arm/overlays/BBAI-PRUIN_PRU1_1.dtbo /lib/firmware/

sudo cp src/arm/overlays/BONE-SPI0_0.dtbo /lib/firmware/

sudo cp src/arm/overlays/BONE-SPI1_0.dtbo /lib/firmware/

make clean 

cd 
nano /boot/uEnv.txt

enable_uboot_overlays=1

uboot_overlay_addr3=/lib/firmware/BBAI-PRUIN_PRU1_1.dtbo

uboot_overlay_addr5=/lib/firmware/BBAI-PRUOUT_PRU1_1.dtbo

sudo reboot

cd /usr/local/sbin
sudo wget -N https://raw.githubusercontent.com/mvduin/bbb-pin-utils/bbai-experimental/show-pins
sudo chmod a+x show-pins

cpufreq-info

cd /etc/init.d
sudo nano cpufrequtils
GOVERNOR="performance"

sudo cpufreq-set -g performance
gcc -o python_call_linux.so -shared -fPIC -O2 python_call.c
