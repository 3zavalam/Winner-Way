import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18

class MultiTaskHead(nn.Module):
    """
    Multi-task head for predicting stroke type and additional metadata
    """
    def __init__(self, in_features, num_stroke_classes=13, 
                 num_hand_classes=2, num_camera_classes=6):
        super(MultiTaskHead, self).__init__()
        
        # Common feature extractor
        self.fc_common = nn.Linear(in_features, 512)
        
        # Task-specific heads
        self.fc_stroke = nn.Linear(512, num_stroke_classes)
        self.fc_hand = nn.Linear(512, num_hand_classes)
        self.fc_camera = nn.Linear(512, num_camera_classes)
    
    def forward(self, x):
        features = F.relu(self.fc_common(x))
        
        stroke_out = self.fc_stroke(features)
        hand_out = self.fc_hand(features)
        camera_out = self.fc_camera(features)
        
        return stroke_out, hand_out, camera_out

class SlowFastNetwork(nn.Module):
    def __init__(self, num_classes=13, num_hand_classes=2, num_camera_classes=6, 
                 alpha=8, beta=1/8, fusion_conv_channel_ratio=2, 
                 multi_task_learning=True):
        super(SlowFastNetwork, self).__init__()
        
        self.multi_task_learning = multi_task_learning
        
        # Slow pathway
        self.slow_pathway = r3d_18(pretrained=True)
        # Modify the first conv layer to take temporal downsampled input
        self.slow_pathway.stem[0] = nn.Conv3d(
            3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False
        )
        
        # Fast pathway
        self.fast_pathway = r3d_18(pretrained=True)
        # Modify channel dimensions according to beta
        fast_inplanes = int(64 * beta)
        self.fast_pathway.stem[0] = nn.Conv3d(
            3, fast_inplanes, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False
        )
        
        # Lateral connections
        self.lateral_connections = nn.ModuleList([
            nn.Conv3d(
                fast_inplanes, int(fusion_conv_channel_ratio * fast_inplanes),
                kernel_size=(5, 1, 1), stride=(alpha, 1, 1), padding=(2, 0, 0), bias=False
            ),
            nn.Conv3d(
                fast_inplanes * 2, int(fusion_conv_channel_ratio * fast_inplanes * 2),
                kernel_size=(5, 1, 1), stride=(alpha, 1, 1), padding=(2, 0, 0), bias=False
            ),
            nn.Conv3d(
                fast_inplanes * 4, int(fusion_conv_channel_ratio * fast_inplanes * 4),
                kernel_size=(5, 1, 1), stride=(alpha, 1, 1), padding=(2, 0, 0), bias=False
            ),
            nn.Conv3d(
                fast_inplanes * 8, int(fusion_conv_channel_ratio * fast_inplanes * 8),
                kernel_size=(5, 1, 1), stride=(alpha, 1, 1), padding=(2, 0, 0), bias=False
            )
        ])
        
        # Replace the final FC layers based on whether using multi-task learning
        if multi_task_learning:
            # Remove the final classifier
            self.slow_pathway.fc = nn.Identity()
            self.fast_pathway.fc = nn.Identity()
            
            # Multi-task heads
            self.slow_multi_task = MultiTaskHead(
                512, num_classes, num_hand_classes, num_camera_classes
            )
            self.fast_multi_task = MultiTaskHead(
                512, num_classes, num_hand_classes, num_camera_classes
            )
        else:
            # Single task (stroke classification only)
            self.slow_pathway.fc = nn.Linear(512, num_classes)
            self.fast_pathway.fc = nn.Linear(512, num_classes)
        
    def forward(self, slow_input, fast_input):
        # Reshape inputs to match expected dimensions
        # Slow pathway expects input of shape (B, C, T, H, W)
        slow_input = slow_input.permute(0, 2, 1, 3, 4)
        # Fast pathway expects input of shape (B, C, T, H, W) 
        fast_input = fast_input.permute(0, 2, 1, 3, 4)
        
        # Forward through slow pathway
        x_slow = self.slow_pathway.stem(slow_input)
        x_fast = self.fast_pathway.stem(fast_input)
        
        # Apply lateral connections after each stage
        layers = [self.slow_pathway.layer1, self.slow_pathway.layer2, 
                  self.slow_pathway.layer3, self.slow_pathway.layer4]
        fast_layers = [self.fast_pathway.layer1, self.fast_pathway.layer2, 
                       self.fast_pathway.layer3, self.fast_pathway.layer4]
        
        for i, (layer, fast_layer, lateral) in enumerate(zip(layers, fast_layers, self.lateral_connections)):
            x_slow = layer(x_slow)
            x_fast = fast_layer(x_fast)
            
            # Apply lateral connection
            lateral_fast = lateral(x_fast)
            x_slow = torch.cat([x_slow, lateral_fast], dim=1)
        
        # Global average pooling
        x_slow = F.adaptive_avg_pool3d(x_slow, (1, 1, 1))
        x_fast = F.adaptive_avg_pool3d(x_fast, (1, 1, 1))
        
        # Flatten
        x_slow = x_slow.view(x_slow.size(0), -1)
        x_fast = x_fast.view(x_fast.size(0), -1)
        
        # Apply final layers (either multi-task or single task)
        if self.multi_task_learning:
            slow_stroke, slow_hand, slow_camera = self.slow_multi_task(x_slow)
            fast_stroke, fast_hand, fast_camera = self.fast_multi_task(x_fast)
            
            # Combine outputs (simple average for now)
            stroke_out = (slow_stroke + fast_stroke) / 2
            hand_out = (slow_hand + fast_hand) / 2
            camera_out = (slow_camera + fast_camera) / 2
            
            return stroke_out, hand_out, camera_out
        else:
            # Single task (stroke classification only)
            x_slow = self.slow_pathway.fc(x_slow)
            x_fast = self.fast_pathway.fc(x_fast)
            
            # Combine outputs (simple average for now)
            out = (x_slow + x_fast) / 2
            
            return out