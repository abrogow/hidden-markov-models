# hidden-markov-models

## Generating `requirements.txt`
```
pipreqs --force .
```

## Generating EXE
```
pyinstaller --onefile --hidden-import=pkg_resources.py2_warn simulator.py
```
The only thing needed to run simulator is `simulator.exe` from `dist` subdirectory and `config` subdirectory with it's contents.
