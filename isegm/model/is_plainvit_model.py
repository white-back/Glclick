import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from isegm.utils.serialization import serialize
from .is_model import ISModel
from .modeling.models_vit import VisionTransformer, PatchEmbed
from .modeling.swin_transformer import SwinTransfomerSegHead
import torchvision.ops.roi_align as roi_align
from isegm.model.ops import DistMaps


class SimpleFPN(nn.Module):
    def __init__(self, in_dim=768, out_dims=[128, 256, 512, 1024]):
        super().__init__()
        self.down_4_chan = max(out_dims[0]*2, in_dim // 2)
        self.down_4 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, self.down_4_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_4_chan),
            nn.GELU(),
            nn.ConvTranspose2d(self.down_4_chan, self.down_4_chan // 2, 2, stride=2),
            nn.GroupNorm(1, self.down_4_chan // 2),
            nn.Conv2d(self.down_4_chan // 2, out_dims[0], 1),
            nn.GroupNorm(1, out_dims[0]),
            nn.GELU()
        )
        self.down_8_chan = max(out_dims[1], in_dim // 2)
        self.down_8 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, self.down_8_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_8_chan),
            nn.Conv2d(self.down_8_chan, out_dims[1], 1),
            nn.GroupNorm(1, out_dims[1]),
            nn.GELU()
        )
        self.down_16 = nn.Sequential(
            nn.Conv2d(in_dim, out_dims[2], 1),
            nn.GroupNorm(1, out_dims[2]),
            nn.GELU()
        )
        self.down_32_chan = max(out_dims[3], in_dim * 2)
        self.down_32 = nn.Sequential(
            nn.Conv2d(in_dim, self.down_32_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_32_chan),
            nn.Conv2d(self.down_32_chan, out_dims[3], 1),
            nn.GroupNorm(1, out_dims[3]),
            nn.GELU()
        )

        self.init_weights()

    def init_weights(self):
        pass

    def forward(self, x):
        x_down_4 = self.down_4(x)
        x_down_8 = self.down_8(x)
        x_down_16 = self.down_16(x)
        x_down_32 = self.down_32(x)

        return [x_down_4, x_down_8, x_down_16, x_down_32]


class PlainVitModel(ISModel):
    @serialize
    def __init__(
        self,
        backbone_params={},
        neck_params={}, 
        head_params={},
        random_split=False,
        **kwargs
        ):

        super().__init__(**kwargs)
        self.random_split = random_split

        self.patch_embed_coords = PatchEmbed(
            img_size= backbone_params['img_size'],
            patch_size=backbone_params['patch_size'], 
            in_chans=3 if self.with_prev_mask else 2, 
            embed_dim=backbone_params['embed_dim'],
        )
        self.patch_embed_coords1 = PatchEmbed(
            img_size= backbone_params['img_size'],
            patch_size=backbone_params['patch_size'], 
            in_chans=3 if self.with_prev_mask else 2, 
            embed_dim=backbone_params['embed_dim'],
        )
        self.patch_embed_coords2 = PatchEmbed(
            img_size= backbone_params['img_size'],
            patch_size=backbone_params['patch_size'], 
            in_chans=3 if self.with_prev_mask else 2, 
            embed_dim=backbone_params['embed_dim'],
        )

        self.backbone = VisionTransformer(**backbone_params)
        self.neck = SimpleFPN(**neck_params)
        self.head = SwinTransfomerSegHead(**head_params)

        self.refiner = RefineLayer(feature_dims=256)
        self.dist_maps_refine = DistMaps(norm_radius=5, spatial_scale=1.0,
                                      cpu_mode=False, use_disks=True)
        self.myfusion = nn.Conv2d(
            in_channels=1024,
            out_channels=256,
            kernel_size=1)
        self.myconv1 = nn.Conv2d(128,256,1)
        self.myconv2 = nn.Conv2d(256,256,1)
        self.myconv3 = nn.Conv2d(512,256,1)
        self.myconv4 = nn.Conv2d(1024,256,1)

    def backbone_forward(self, image, coord_features=None):
        coord_feats = []
        coord_feats.append(self.patch_embed_coords1(coord_features))
        coord_feats.append(self.patch_embed_coords2(coord_features))
        coord_features = self.patch_embed_coords(coord_features)
        backbone_features = self.backbone.forward_backbone(image, coord_features, coord_feats,self.random_split)

        # Extract 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        B, N, C = backbone_features.shape
        grid_size = self.backbone.patch_embed.grid_size

        backbone_features = backbone_features.transpose(-1,-2).view(B, C, grid_size[0], grid_size[1])
        multi_scale_features = self.neck(backbone_features)

        return {'feature':multi_scale_features,'instances': self.head(multi_scale_features)['instances'], 'instances_aux': None}

    def refine(self, cropped_image, cropped_points, full_feature, full_logits, bboxes):
        '''
        bboxes : [b,5]
        '''
        h1 = cropped_image.shape[-1]
        h20 = full_feature[0].shape[-1]
        h21 = full_feature[1].shape[-1]
        h22 = full_feature[2].shape[-1]
        h23 = full_feature[3].shape[-1]
        r0 = h1/h20
        r1 = h1/h21
        r2 = h1/h22
        r3 = h1/h23

        cropped_feature = []
        cropped_feature.append(roi_align(full_feature[0],bboxes,full_feature[0].size()[2:], spatial_scale=1/r0, aligned = True))
        cropped_feature.append(roi_align(full_feature[1],bboxes,full_feature[1].size()[2:], spatial_scale=1/r1, aligned = True))
        cropped_feature.append(roi_align(full_feature[2],bboxes,full_feature[2].size()[2:], spatial_scale=1/r2, aligned = True))
        cropped_feature.append(roi_align(full_feature[3],bboxes,full_feature[3].size()[2:], spatial_scale=1/r3, aligned = True))

        cropped_feature[0] = resize(input=self.myconv1(cropped_feature[0]),size=cropped_feature[0].shape[2:])
        cropped_feature[1] = resize(input=self.myconv2(cropped_feature[1]),size=cropped_feature[0].shape[2:])
        cropped_feature[2] = resize(input=self.myconv3(cropped_feature[2]),size=cropped_feature[0].shape[2:])
        cropped_feature[3] = resize(input=self.myconv4(cropped_feature[3]),size=cropped_feature[0].shape[2:])


        cropped_feature = self.myfusion(torch.cat(cropped_feature, dim=1))

        cropped_logits = roi_align(full_logits,bboxes,cropped_image.size()[2:], spatial_scale=1, aligned = True)
        
        click_map = self.dist_maps_refine( cropped_image,cropped_points)
        refined_mask, trimap = self.refiner(cropped_image,click_map,cropped_feature,cropped_logits)
        return {'instances_refined': refined_mask, 'instances_coarse':cropped_logits, 'trimap':trimap}

    def forward(self, image, points):
        image, prev_mask = self.prepare_input(image)
        coord_features = self.get_coord_features(image, prev_mask, points)
        click_map = coord_features[:,1:,:,:]

        
        small_image = image
        small_coord_features = coord_features

        small_coord_features = self.maps_transform(small_coord_features)
        outputs = self.backbone_forward( small_image, small_coord_features)

        outputs['click_map'] = click_map
        outputs['instances'] = nn.functional.interpolate(outputs['instances'], size=image.size()[2:],
                                                         mode='bilinear', align_corners=True)
        if self.with_aux_output:
            outputs['instances_aux'] = nn.functional.interpolate(outputs['instances_aux'], size=image.size()[2:],
                                                             mode='bilinear', align_corners=True)
        return outputs
    
class RefineLayer(nn.Module):
    """
    Refine Layer for Full Resolution
    """
    def __init__(self, input_dims = 6, mid_dims = 32, feature_dims = 96, num_classes = 1,  **kwargs):
        super(RefineLayer, self).__init__()
        self.num_classes = num_classes
        self.image_conv1 = ConvModule(
            in_channels=input_dims,
            out_channels= mid_dims,
            kernel_size=3,
            stride=2,
            padding=1,
            #norm_cfg=dict(type='BN', requires_grad=True),
            )
        self.image_conv2 = XConvBnRelu( mid_dims, mid_dims)
        self.image_conv3 = XConvBnRelu( mid_dims, mid_dims)
        

        self.refine_fusion = ConvModule(
            in_channels= feature_dims,
            out_channels= mid_dims,
            kernel_size=1,
            stride=1,
            padding=0,
            #norm_cfg=dict(type='BN', requires_grad=True),
            )
        
        self.refine_fusion2 = XConvBnRelu( mid_dims, mid_dims)
        self.refine_fusion3 = XConvBnRelu( mid_dims, mid_dims)
        self.refine_pred = nn.Conv2d( mid_dims, num_classes,3,1,1)
        self.refine_trimap = nn.Conv2d( mid_dims, num_classes,3,1,1)

    

    def forward(self, input_image, click_map, final_feature, cropped_logits):
        #cropped_logits = cropped_logits.clone().detach()
        #final_feature = final_feature.clone().detach()

        mask = cropped_logits #resize(cropped_logits, size=input_image.size()[2:],mode='bilinear',align_corners=True)
        bin_mask = torch.sigmoid(mask) #> 0.49
        input_image = torch.cat( [input_image,click_map,bin_mask], 1)

        final_feature = self.refine_fusion(final_feature)
        image_feature = self.image_conv1(input_image)
        image_feature = self.image_conv2(image_feature)
        image_feature = self.image_conv3(image_feature)
        pred_feature = resize(final_feature, size=image_feature.size()[2:],mode='bilinear',align_corners=True)
        #fusion_gate = self.feature_gate(final_feature)
        #fusion_gate = resize(fusion_gate, size=image_feature.size()[2:],mode='bilinear',align_corners=True)
        pred_feature = pred_feature + image_feature #* fusion_gate
        

        pred_feature = self.refine_fusion2(pred_feature)
        pred_feature = self.refine_fusion3(pred_feature)
        pred_full = self.refine_pred(pred_feature)
        trimap = self.refine_trimap(pred_feature)
        trimap = F.interpolate(trimap, size=input_image.size()[2:],mode='bilinear',align_corners=True)
        pred_full = F.interpolate(pred_full, size=input_image.size()[2:],mode='bilinear',align_corners=True)
        trimap_sig = torch.sigmoid(trimap)
        pred = pred_full * trimap_sig + mask * (1-trimap_sig)
        return pred, trimap

class ConvModule(nn.Module):
    def __init__(self, in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
       

    def forward(self, x):
        return self.activation( self.norm( self.conv(x)  ) )




class XConvBnRelu(nn.Module):
    """
    Xception conv bn relu
    """
    def __init__(self, input_dims = 3, out_dims = 16,   **kwargs):
        super(XConvBnRelu, self).__init__()
        self.conv3x3 = nn.Conv2d(input_dims,input_dims,3,1,1,groups=input_dims)
        self.conv1x1 = nn.Conv2d(input_dims,out_dims,1,1,0)
        self.norm = nn.BatchNorm2d(out_dims)
        self.activation = nn.ReLU()
    def forward(self,x):
        x = self.conv3x3(x)
        x = self.conv1x1(x)
        x = self.norm(x)
        x = self.activation(x)
        return x




def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           ):
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)