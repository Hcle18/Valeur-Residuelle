'''
This file is used to manage images in the application.
'''

# Package imports
import base64
import os

LOGO_NAVBAR = '1. NexiaLog_RVB_HD_Original - numérique PNG.png'
ICONE_SIDEBAR = 'NexiaLog_RVB_Original.png'

cwd = os.getcwd()

# logo information
logo_path = os.path.join(cwd, 'src', 'app', 'assets', LOGO_NAVBAR)
logo_tunel = base64.b64encode(open(logo_path, 'rb').read())
logo_encoded = 'data:image/png;base64,{}'.format(logo_tunel.decode())

# Icone
icon_path = os.path.join(cwd, 'src', 'app', 'assets', ICONE_SIDEBAR)
icon_tunel = base64.b64encode(open(icon_path, 'rb').read())
icon_encoded = 'data:image/png;base64,{}'.format(icon_tunel.decode())

