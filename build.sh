cd gym_microrts/microrts


rm -rf build microrts.jar
mkdir build
javac -d "./build" -cp "./lib/*;./lib/jena/*" -sourcepath "./src"  $(find ./src/* | grep .java)
cp -a lib/. build/
cp -a lib/jena/. build/jena/

# hack to remove the weka dependency in build time
# we don't use weka anyway yet it's a 10 MB package
rm build/weka.jar
rm -rf build/bots

cd build
for i in *.jar; do
    echo "adding dependency $i"
    jar xf $i
done
jenaSubsystems=""
for i in jena/*.jar; do
    echo "adding jena dependency $i"
    jar xf $i
    if [ -f META-INF/services/org.apache.jena.sys.JenaSubsystemLifecycle ]; then
        printf -v jenaSubsystems '%s%s\n' "${jenaSubsystems}" "$(cat META-INF/services/org.apache.jena.sys.JenaSubsystemLifecycle)"
    fi
done
touch META-INF/services/org.apache.jena.sys.JenaSubsystemLifecycle
echo "$jenaSubsystems" > META-INF/services/org.apache.jena.sys.JenaSubsystemLifecycle
jar cvf microrts.jar *
mv microrts.jar ../microrts.jar
cd ..
rm -rf build