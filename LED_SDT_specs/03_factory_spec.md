# 03 - Update `networks/net_factory.py`

## Current code

```python
from networks.unet import UNet, UNet_HL

def net_factory(net_type="unet", in_chns=1, class_num=3):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_hl":
        net = UNet_HL(in_chns=in_chns, class_num=class_num).cuda()
    else:
        net = None
    return net
```

## Required modification

Add import:

```python
from networks.unet_lgdt import UNet_LGDT
```

Add branch:

```python
elif net_type == "unet_lgdt":
    net = UNet_LGDT(in_chns=in_chns, class_num=class_num).cuda()
```

## Expected final

```python
from networks.unet import UNet, UNet_HL
from networks.unet_lgdt import UNet_LGDT

def net_factory(net_type="unet", in_chns=1, class_num=3):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_hl":
        net = UNet_HL(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_lgdt":
        net = UNet_LGDT(in_chns=in_chns, class_num=class_num).cuda()
    else:
        raise ValueError(f"Unknown net_type: {net_type}")
    return net
```

Use `raise ValueError` rather than returning `None`.
