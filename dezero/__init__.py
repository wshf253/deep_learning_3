# True: ~step32
is_simple_core = False # True

if is_simple_core:
    from dezero.core_simple import Variable
    from dezero.core_simple import Function
    from dezero.core_simple import using_config
    from dezero.core_simple import no_grad
    from dezero.core_simple import as_array
    from dezero.core_simple import as_variable
    from dezero.core_simple import setup_varibale
else:
    from dezero.core import Variable
    from dezero.core import Parameter
    from dezero.core import Function
    from dezero.core import using_config
    from dezero.core import no_grad
    from dezero.core import test_mode
    from dezero.core import as_array
    from dezero.core import as_variable
    from dezero.core import setup_varibale
    from dezero.core import Config
    from dezero.layers import Layer
    from dezero.models import Model
    from dezero.datasets import Dataset
    from dezero.dataloaders import DataLoader

    import dezero.datasets
    import dezero.dataloaders
    import dezero.optimizers
    import dezero.functions
    import dezero.layers
    import dezero.utils
    import dezero.cuda
    import dezero.transforms

setup_varibale()