import argparse
import config
import inspect
import sys

def create_dynamic_parser():
    """Create argparse parser that discovers all config parameters automatically"""
    parser = argparse.ArgumentParser(
        description='Speech Encoding Analysis - Dynamic Configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --bands Alpha Theta --default_alpha 100
  python main.py --sessions 21 22 23 --n_folds 25
  python main.py --stimuli Envelope --statistical_test
        """
    )
    
    # Discover all config attributes dynamically
    for attr_name in dir(config):
        if not attr_name.startswith('_'):  # Skip private attributes
            attr_value = getattr(config, attr_name)
            
            # Skip modules, functions, and classes
            if inspect.ismodule(attr_value) or inspect.isfunction(attr_value) or inspect.isclass(attr_value):
                continue
            
            # Add argument based on type
            if attr_value is None:
                # Example: treat as float if you expect a float
                if attr_name in ["temporal_shift", "set_alpha"]:
                    parser.add_argument(f'--{attr_name}', type=float, default=None, help=f'(float/None) {attr_name}')
            elif isinstance(attr_value, bool):
                # For booleans, use store_true/store_false but set default to a sentinel value
                parser.add_argument(f'--{attr_name}', action='store_true', default=argparse.SUPPRESS,
                                  help=f'Enable {attr_name} (current: {attr_value})')
                parser.add_argument(f'--no-{attr_name}', action='store_false', dest=attr_name, default=argparse.SUPPRESS,
                                  help=f'Disable {attr_name} (current: {attr_value})')
            elif isinstance(attr_value, list):
                parser.add_argument(f'--{attr_name}', nargs='*', default=argparse.SUPPRESS,
                                  help=f'Set {attr_name} (current: {attr_value})')
            elif isinstance(attr_value, (int, float, str)):
                parser.add_argument(f'--{attr_name}', type=type(attr_value), default=argparse.SUPPRESS,
                                  help=f'Set {attr_name} (current: {attr_value})')
    
    return parser

def apply_args_to_config(args):
    """Apply parsed arguments to config"""
    for arg_name, arg_value in vars(args).items():
        if hasattr(config, arg_name):
            current_value = getattr(config, arg_name)
            
            # Handle type conversion for lists
            if isinstance(current_value, list) and arg_value:
                if current_value and isinstance(current_value[0], int):
                    arg_value = [int(x) for x in arg_value]
                elif current_value and isinstance(current_value[0], float):
                    arg_value = [float(x) for x in arg_value]
            
            setattr(config, arg_name, arg_value)
            print(f"Override: {arg_name} = {arg_value}")

# Use it
parser = create_dynamic_parser()
args = parser.parse_args()
apply_args_to_config(args)