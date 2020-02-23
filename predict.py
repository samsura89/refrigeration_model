import getopt
import sys

try:
    options, remainder = getopt.gnu_getopt(
        sys.argv[1:],
        's:u',
        ['setpoint=',
         'uuid=',
         ])
except getopt.GetoptError as err:
    print('ERROR:', err)
    sys.exit(1)

for opt, arg in options:
    if opt in '--setpoint':
        setpoint_arg = arg
    elif opt == '--uuid':
        uuid_arg = arg

#call approprietely
# load_model(uuid=uuid_arg, setpoint=setpoint_arg)

