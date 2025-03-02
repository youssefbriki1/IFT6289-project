conda env export | grep -v "^prefix: " > environment.yml
sed -n '/pip:/,$p' environment.yml | tail -n +2 | sed 's/      - //g' > requirements.txt
