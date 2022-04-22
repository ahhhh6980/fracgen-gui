pname="$(basename "$PWD")"

cargo build --release
cargo build --release --target x86_64-pc-windows-gnu

echo "Enter a version"
read version

linux="target/release/${pname}"
linuxbuild="binaries/${pname}_${version}_linux_x86-64"

windows="target/x86_64-pc-windows-gnu/release/${pname}.exe"
windowsbuild="binaries/${pname}_${version}_windows_x86-64"

mkdir -p $linuxbuild
mkdir -p $windowsbuild

cp $linux $linuxbuild
cp $windows $windowsbuild

for file in "JoeBiden-2.webp"
do
	cp -r $file $linuxbuild
	cp -r $file $windowsbuild
done

cd binaries

linuxbuild="${pname}_${version}_linux_x86-64"
windowsbuild="${pname}_${version}_windows_x86-64"

zip -r "${windowsbuild}.zip" $windowsbuild
tar -czf "${linuxbuild}.tar.gz" ${linuxbuild}
