#!/usr/bin/env python3

import connexion

from server import encoder


def main():
    app = connexion.App(__name__, specification_dir='./swagger/')
    app.app.json_encoder = encoder.JSONEncoder
    app.add_api('swagger.yaml', arguments={'title': 'NI SFC Sub-Module Service'})
    app.run(port=8001)


if __name__ == '__main__':
    main()
