from torch.nn import functional as F
from models.SEB.hyp_crossvit import *
from models.SEB.mobilefacenet import MobileFaceNet
from models.SEB.ir50 import Backbone


def load_pretrained_weights(model, checkpoint):
    import collections
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    # new_state_dict.requires_grad = False
    model_dict.update(new_state_dict)

    model.load_state_dict(model_dict)
    # print('load_weight', len(matched_layers))
    return model


class SE_block(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
        self.linear2 = torch.nn.Linear(input_dim, input_dim)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x1 = self.linear1(x)
        x1 = self.relu(x1)
        x1 = self.linear2(x1)
        x1 = self.sigmod(x1)
        x = x * x1
        return x


class StaticModel(nn.Module):
    def __init__(self, type="small"):
        super().__init__()
        depth = 4
        if type == "small":
            depth = 4
        # if type == "base":
        #     depth = 6
        # if type == "large":
        #     depth = 8

        self.face_landback = MobileFaceNet([112, 112],136)
        face_landback_checkpoint = torch.load('C:\\Users\\DELL\\Desktop\\MultiFPE\\ckpt\\mobilefacenet_model_best.pth.tar', map_location=lambda storage, loc: storage, weights_only=True)
        self.face_landback.load_state_dict(face_landback_checkpoint['state_dict'])

        for param in self.face_landback.parameters():
            param.requires_grad = False

        ###########################################################################333

        self.ir_back = Backbone(50, 0.0, 'ir')
        ir_checkpoint = torch.load('C:\\Users\\DELL\\Desktop\\MultiFPE\\ckpt\\ir50.pth', map_location=lambda storage, loc: storage, weights_only=True)
        self.ir_back = load_pretrained_weights(self.ir_back, ir_checkpoint)
        for param in self.ir_back.parameters():
            param.requires_grad = False

        self.ir_layer = nn.Linear(1024,512)

        #############################################################3

        self.pyramid_fuse = HyVisionTransformer(in_chans=49, q_chanel = 49, embed_dim=512,
                                             depth=depth, num_heads=4, mlp_ratio=2.,
                                             drop_rate=0.4, attn_drop_rate=0.4, drop_path_rate=0.3)


        self.se_block = SE_block(input_dim=512)

    def forward(self, static_image, mask=None):
        device = next(self.parameters()).device

        static_image = static_image.to(device)

        x = static_image
        x = x.squeeze(1)
        B_ = x.shape[0]
        x_face = F.interpolate(x, size=112)
        _, x_face = self.face_landback(x_face)
        x_face = x_face.view(B_, -1, 49).transpose(1,2)
        ###############  landmark x_face ([B, 49, 512])

        x_ir = self.ir_back(x)
        x_ir = self.ir_layer(x_ir)
        ###############  image x_ir ([B, 49, 512])

        y_hat = self.pyramid_fuse(x_ir, x_face)
        y_hat = self.se_block(y_hat)
        out = y_hat
        return out


