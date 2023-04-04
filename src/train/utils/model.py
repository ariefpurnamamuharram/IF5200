from torch import nn


def build_model(
        model,
        d_class: int = 2,
        fine_tuning: bool = True,
        model_info: bool = False,
        params_info: bool = False):
    # At least 2 class!
    if d_class <= 1:
        raise ValueError('Can not less than 2 classes!')

    # Setup params grad
    requires_grad = True if (fine_tuning) else False
    for param in model.parameters():
        param.requires_grad = requires_grad

    # Setup classification layer
    model.fc = nn.LazyLinear(d_class)

    # Print model info
    if model_info:
        print('Model Info')
        print('-' * 13)
        print(model)

    print('\n')

    # Print params info
    if params_info:
        print('Params Info')
        print('-' * 13)
        for name, param in model.named_parameters():
            print(name, param.requires_grad)

    # Return the model
    return model
