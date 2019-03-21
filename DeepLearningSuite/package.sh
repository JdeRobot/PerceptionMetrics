#!/usr/bin/env bash

########################################################################
# Package the binaries built on Travis-CI as an AppImage
# By Simon Peter 2016
# For more information, see http://appimage.org/
########################################################################

export ARCH=$(arch)

if [[ "$TO_TEST" == "WITH_ROS_AND_ICE" ]];
then
APP=DetectionSuite_with_ROS_and_ICE
else
APP=DetectionSuite
fi

LOWERAPP=${APP,,}


mkdir -p $APP.AppDir/usr/

cd $APP.AppDir

echo `pwd`

mkdir -p usr/bin
cp ../DatasetEvaluationApp/DatasetEvaluationApp usr/bin/

mkdir -p usr/lib
ldd ../DatasetEvaluationApp/DatasetEvaluationApp | grep "=> /" | awk '{print $3}' | xargs -I '{}' cp -v '{}' usr/lib/

echo "Now copying Qt plugin libraries"
mkdir usr/bin/platforms/

# For Qt Dependency
cp -v `find /usr -iname 'libqxcb.so'` usr/bin/platforms

find /usr -iname 'libqxcb.so' | xargs ldd | grep "=> /" | awk '{print $3}' | xargs -I '{}' cp -v '{}' usr/bin/platforms

# Copying necessary python modules
cp -v -r ../../DeepLearningSuiteLib/python_modules usr/lib/

cd usr/ ; find . -type f -exec sed -i -e 's|/usr|././|g' {} \; ; cd -

cat > AppRun << 'EOF'
#!/usr/bin/env bash
# some magic to find out the real location of this script dealing with symlinks
DIR=`readlink "$0"` || DIR="$0";
DIR=`dirname "$DIR"`;
cd "$DIR"
DIR=`pwd`
cd - > /dev/null
# disable parameter expansion to forward all arguments unprocessed to the VM
set -f
# run the VM and pass along all arguments as is
export PYTHONPATH="$DIR/usr/lib/python_modules"
LD_LIBRARY_PATH="$DIR/usr/lib" "${DIR}/usr/bin/DatasetEvaluationApp" "$@"
EOF

chmod +x AppRun

wget http://files.pharo.org/media/logo/icon-lighthouse-512x512.png -O $APP.png

cat > $APP.desktop <<EOF
[Desktop Entry]
Name=$APP
Icon=$APP
Exec=AppRun
Type=Application
Terminal=true
Categories=Education;
EOF

cd ..

wget "https://github.com/probonopd/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage"
chmod a+x appimagetool-x86_64.AppImage

echo `pwd`

mkdir out
cd out 

../appimagetool-x86_64.AppImage ../$APP.AppDir

cd .. # Since this is being run in the same shell it is necssary to go backwards
