import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CSDN_Tem(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CSDN_Tem, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class ZeroDCEPlus(nn.Module):
    def __init__(self):
        super(ZeroDCEPlus, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        number_f = 32
        self.e_conv1 = CSDN_Tem(3, number_f)
        self.e_conv2 = CSDN_Tem(number_f, number_f)
        self.e_conv3 = CSDN_Tem(number_f, number_f)
        self.e_conv4 = CSDN_Tem(number_f, number_f)
        self.e_conv5 = CSDN_Tem(number_f * 2, number_f)
        self.e_conv6 = CSDN_Tem(number_f * 2, number_f)
        self.e_conv7 = CSDN_Tem(number_f * 2, 3)

    def enhance(self, x, x_r):
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        enhance_image_1 = x + x_r * (torch.pow(x, 2) - x)
        x = enhance_image_1 + x_r * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        enhance_image = x + x_r * (torch.pow(x, 2) - x)
        return enhance_image

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        enhance_image = self.enhance(x, x_r)
        return enhance_image, x_r

class AdaptiveLowLightEnhancer:
    def __init__(self, device="cuda", brightness_thresh=0.4, variance_thresh=0.05, weights_path='Epoch99.pth'):
        self.device = device
        self.brightness_thresh = float(brightness_thresh)
        self.variance_thresh = variance_thresh
        self.model = ZeroDCEPlus().to(self.device)
        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

    def _is_dark(self, frame_bgr: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        mean_brightness = gray.mean() / 255.0
        variance = gray.var() / 255.0
        return mean_brightness < self.brightness_thresh and variance > self.variance_thresh

    def enhance(self, frame_bgr: np.ndarray, apply_check: bool = True) -> tuple:
        if apply_check and not self._is_dark(frame_bgr):
            return frame_bgr, False
        frame_tensor = torch.from_numpy(frame_bgr).float().to(self.device)
        frame_tensor = frame_tensor / 255.0
        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            enhanced_tensor, _ = self.model(frame_tensor)
        enhanced_frame = enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        enhanced_frame = (enhanced_frame * 255).astype(np.uint8)
        lab = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        enhanced_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return enhanced_frame, True