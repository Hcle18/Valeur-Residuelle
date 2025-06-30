'''
This file is used to manage images in the application.
'''

# Package imports
import base64
import os

LOGO_NAVBAR = '1. NexiaLog_RVB_HD_Original - numérique PNG.png'

# logo information
cwd = os.getcwd()
logo_path = os.path.join(cwd, 'src', 'app', 'assets', LOGO_NAVBAR)
logo_tunel = base64.b64encode(open(logo_path, 'rb').read())
logo_encoded = 'data:image/png;base64,{}'.format(logo_tunel.decode())