"""Bump the package version to the current datetime.

This script edits the TOML file and puts the current datetime as the package
version.
"""

import re
import time
from tomlkit.toml_file import TOMLFile

TOML_FILE = "pyproject.toml"

def run():
    toml_file = TOMLFile(TOML_FILE)
    t = toml_file.read()

    version = t['tool']['poetry']['version']
    new_minor = time.strftime("%Y%m%d%H%M")
    # This is not very conformant, but I couldn't find utils to set version.
    version = re.sub('\.[^.]*$', '.' + new_minor, version)
    t['tool']['poetry']['version'] = version

    name = t['tool']['poetry']['name']
    name += '-dev'
    t['tool']['poetry']['name'] = name

    t['tool']['poetry']['description'] += '  Dev package that pushes on green from master.'

    toml_file.write(t)

if __name__ == '__main__':
    run()
