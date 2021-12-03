
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1BCxmqLANXHbBaWs8A7_jqfVUv8mydp5R' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1BCxmqLANXHbBaWs8A7_jqfVUv8mydp5R" -O miniimagenet.zip && rm -rf /tmp/cookies.txt

unzip miniimagenet.zip miniimagenet/
