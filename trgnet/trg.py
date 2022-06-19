from trgnet.backbones.mobilenetv2 import MobileNetV2


backbone = MobileNetV2()
print(backbone.eval())
print(sum(p.numel() for p in backbone.parameters() if p.requires_grad))
