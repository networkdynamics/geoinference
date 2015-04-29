#!/bin/bash

# assume that the user has the following installed with apt-get
# GDAL/OGR 1.8+ (not needed if fiona is not needed)

# install all dependencies 
# list package names here
array=(numpy scipy nltk haversine geographiclib tinysegmenter pyshp shapely geopy requests)

# use -u to specify that the packages are installed for user
# use -s, installed with sudo

USER=""
SUDO=""
DIR="./"

options='usdh'
while getopts $options option
do
	case $option in
		u  )    USER=" --user";;
	        s  )    SUDO="sudo ";;
		h  )	echo "USAGE: ${0} [-u or -s] [dir]\n
	-u to sepcify that the packages are intalled for user only\n
	-s to specify that the packages are installed with sudo\n
	[dir] the directory to save downloaded packages from github";exit 0;;
	esac
done

shift $((OPTIND - 1))

if [ -n "$1" ]; then
	DIR=$1
fi

for PACK in "${array[@]}"; do
        ${SUDO}pip install${USER} ${PACK} --upgrade
	echo $PACK
done

# installing scikit
${SUDO} pip install${USER} --install-option="--prefix=" -U scikit-learn

# installing zen
echo "~~~ Installing zen. Package will be save under directory ${DIR}"
git clone git://github.com/networkdynamics/zenlib.git ${DIR}
cd ${DIR}/zenlib/src 
python setup.py install
cd ../
python -m zen.test`u
