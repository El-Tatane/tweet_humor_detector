# Installation

Installer docker et docker-compose (Linux, Windows, Mac)


# Get starting

``` bash
    cd docker
    cp .env.dist .env
```

Modifier le fichier .env sur le chemin du repertoire des données (DATA_PATH)


``` bash
   docker-compose build
   docker-compose up -d
```

Se rendre sur le port 9999 (ou valeur de la varible "JUPYTER_PORT" dans le fichier .env)