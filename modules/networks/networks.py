from modules.networks import resnet, simple_network


network_names = {
    "resnet18": resnet.ResNet18,
    "resnet34": resnet.ResNet34,
    "resnet44": resnet.ResNet44,
    "resnet50": resnet.ResNet50,
    "simple": simple_network.SimpleNetwork,
    "z2cnn": simple_network.Z2CNN,
    "hybrid_z2cnn": simple_network.HybridZ2CNN,
}

def parse_name(network_name):
    """
    Returns the network class based on the network name.
    """

    if network_name not in network_names:
        raise ValueError(f"Network name not recognized: {network_name}. Available networks are {list(network_names.keys())}")

    return network_names[network_name]