import os

import torch
import numpy as np
import SimpleITK as sitk
import random 

from torch.utils.data import Dataset

class BraTS(Dataset):
    def __init__(self, root, phase, desired_depth=128, desired_height=160, desired_width=192, normalize_flag=True, 
                 scale_intensity_flag=False, shift_intesity_flag=False, flip_axes_flag=False):
        
        self.root = root
        self.patients = os.listdir(self.root)
        self.patients = [x for x in self.patients if x.startswith('BraTS')]
        self.flair_suffix = "_flair.nii.gz"
        self.t1_suffix = "_t1.nii.gz"
        self.t1ce_suffix = "_t1ce.nii.gz"
        self.t2_suffix = "_t2.nii.gz"
        self.seg_suffix = "_seg.nii.gz"
        self.wt_suffix = "_contour_wt.nii.gz"
        self.tc_suffix = "_contour_tc.nii.gz"
        self.et_suffix = "_contour_et.nii.gz"
        self.phase = phase
        self.desired_depth = desired_depth
        self.desired_height = desired_height
        self.desired_width = desired_width
        self.normalize_flag = normalize_flag
        self.scale_intensity_flag = scale_intensity_flag
        self.shift_intensity_flag = shift_intesity_flag
        self.flip_axes_flag = flip_axes_flag
        
    def __len__(self):
        return len(self.patients)
        
    def __getitem__(self, idx):
        patient = self.patients[idx]
        path_flair = os.path.join(self.root, patient, patient + self.flair_suffix)
        path_t1 = os.path.join(self.root, patient, patient + self.t1_suffix)
        path_t2 = os.path.join(self.root, patient, patient + self.t2_suffix)
        path_t1ce = os.path.join(self.root, patient, patient + self.t1ce_suffix)
        path_seg = os.path.join(self.root, patient, patient + self.seg_suffix)
        
        path_contour_wt = os.path.join(self.root, patient, patient + self.wt_suffix)
        path_contour_tc = os.path.join(self.root, patient, patient + self.tc_suffix)
        path_contour_et = os.path.join(self.root, patient, patient + self.et_suffix)
        
        mask, start_depth, start_height, start_width = self.get_mask_simple(path_seg)
        
        out = self.get_volume(path_flair, path_t1, path_t2, path_t1ce, start_depth, 
                                                                      start_height, start_width)
        contours = self.get_contours(path_contour_wt, path_contour_tc, path_contour_et, start_depth, start_height, start_width)
        
        if self.flip_axes_flag:
            dice = random.uniform(0, 1)
            if dice > 0.5 and dice < 0.6:
                mask = mask[:, ::-1, : , :].copy()
                out = out[:, ::-1, : , :].copy()
                contours = contours[:, ::-1, : , :].copy()
            elif dice > 0.6 and dice < 0.7:
                mask = mask[:, :, ::-1 , :].copy()
                out = out[:, :, ::-1 , :].copy()
                contours = contours[:, :, ::-1 , :].copy()         
            elif dice > 0.7 and dice < 0.8:
                mask = mask[:, :, : , ::-1].copy()
                out = out[:, :, : , ::-1].copy()
                contours = contours[:, :, : , ::-1].copy()
            elif dice > 0.8 and dice < 0.9:
                mask = mask[:, :, ::-1 , ::-1].copy()
                out = out[:, :, ::-1 , ::-1].copy()
                contours = contours[:, :, ::-1 , ::-1].copy()
            elif dice > 0.9 and dice < 1:
                mask = mask[:, ::-1, ::-1 , ::-1].copy()
                out = out[:, ::-1, ::-1 , ::-1].copy()
                contours = contours[:, ::-1, ::-1 , ::-1].copy()
        
        return torch.FloatTensor(out), torch.FloatTensor(mask), torch.FloatTensor(contours), patient
    
    def get_contours(self, path_contour_wt, path_contour_tc, path_contour_et, start_depth, start_height, start_width):
        depth = self.desired_depth
        height = self.desired_height
        width = self.desired_width
        
        try:
            contour_wt = sitk.GetArrayFromImage(sitk.ReadImage(path_contour_wt))[start_depth: start_depth + depth, start_height: start_height + height, start_width: start_width + width]
        except Exception as e:
            return np.zeros((self.desired_depth, self.desired_height, self.desired_width))
        
        try:
            contour_tc = sitk.GetArrayFromImage(sitk.ReadImage(path_contour_tc))[start_depth: start_depth + depth, start_height: start_height + height, start_width: start_width + width]
        except Exception as e:
            return np.zeros((self.desired_depth, self.desired_height, self.desired_width))
        
        try:
            contour_et = sitk.GetArrayFromImage(sitk.ReadImage(path_contour_et))[start_depth: start_depth + depth, start_height: start_height + height, start_width: start_width + width]
        except Exception as e:
            return np.zeros((self.desired_depth, self.desired_height, self.desired_width))
        
        
        return np.stack((contour_wt, contour_tc, contour_et))
        
    
    def normalize_one_volume(self, volume):
        new_volume = np.zeros(volume.shape)
        location = np.where(volume != 0)
        mean = np.mean(volume[location])
        var = np.std(volume[location])
        new_volume[location] = (volume[location] - mean) / var

        return new_volume

    def merge_volumes(self, *volumes):
        return np.stack(volumes, axis=0)

    def shift_intensity(self, volume):
        location = np.where(volume != 0)
        minimum = np.min(volume[location])
        maximum = np.max(volume[location])
        std = np.std(volume[location])
        value = np.random.uniform(low=-0.1 * std, high=0.1 * std, size=1)
        volume[location] += value
        volume[location][volume[location] < minimum] = minimum
        volume[location][volume[location] > maximum] = maximum

        return volume

    def scale_intensity(self, volume):
        location = np.where(volume != 0)
        new_volume = np.zeros(volume.shape)
        IntensityScale = np.random.uniform(0.9, 1, 1)
        new_volume[location] = volume[location] * IntensityScale
        return new_volume

    def crop_volume(self, volume, start_depth, start_height, start_width):
        ### initial volumes are 155 X 240 X 240
        depth = self.desired_depth
        height = self.desired_height
        width = self.desired_width
        
        
        return volume[:, start_depth: start_depth + depth, start_height: start_height + height, start_width: start_width + width]
        
    def get_volume(self, path_flair, path_t1, path_t2, path_t1ce, start_depth, start_height, start_width):
        flair = sitk.GetArrayFromImage(sitk.ReadImage(path_flair))
        t1 = sitk.GetArrayFromImage(sitk.ReadImage(path_t1))
        t2 = sitk.GetArrayFromImage(sitk.ReadImage(path_t2))
        t1ce = sitk.GetArrayFromImage(sitk.ReadImage(path_t1ce))
        
        if self.desired_depth > 155:
            flair = np.concatenate([flair, np.zeros((self.desired_depth - 155, 240, 240))], axis=0)
            t1 = np.concatenate([t1, np.zeros((self.desired_depth - 155, 240, 240))], axis=0)
            t2 = np.concatenate([t2, np.zeros((self.desired_depth - 155, 240, 240))], axis=0)
            t1ce = np.concatenate([t1ce, np.zeros((self.desired_depth - 155, 240, 240))], axis=0)
            
        if self.scale_intensity_flag:
            flair = self.scale_intensity(flair)
            t1 = self.scale_intensity(t1)
            t2 = self.scale_intensity(t2)
            t1ce = self.scale_intensity(t1ce)
            
        if self.shift_intensity_flag:
            flair = self.shift_intensity(flair)
            t1 = self.shift_intensity(t1)
            t2 = self.shift_intensity(t2)
            t1ce = self.shift_intensity(t1ce)
            
        if self.normalize_flag == True:
            out = self.merge_volumes(self.normalize_one_volume(flair), self.normalize_one_volume(t2), self.normalize_one_volume(t1ce), 
                                self.normalize_one_volume(t1))
        else:
            out = self.merge_volumes(flair, t2, t1ce, t1)

        out = self.crop_volume(out, start_depth, start_height, start_width)
        return out
    
    def get_mask_simple(self, path_seg):
        try:
            seg = sitk.GetArrayFromImage(sitk.ReadImage(path_seg))
        except Exception as e:
            seg = np.zeros((155, 240, 240))
        
        desired_depth = self.desired_depth
        desired_height = self.desired_height
        desired_width = self.desired_width

        if desired_depth <= 155:
            start_depth = np.random.randint(0, 156 - desired_depth)
            to_add = 0
        else:
            start_depth = 0
            to_add = desired_depth - 155
            desired_depth = 155
        
        start_height = np.random.randint(0, 241 - desired_height)
        start_width = np.random.randint(0, 241 - desired_width)
        
        end_depth = start_depth + desired_depth
        end_height = start_height + desired_height
        end_width = start_width + desired_width
        
        if to_add != 0:
            pad_seg = np.zeros((to_add, end_height - start_height, end_width - start_width))
        new_seg = seg[start_depth: end_depth, start_height: end_height, start_width: end_width]
        
        if to_add != 0:
            new_seg = np.concatenate([new_seg, pad_seg], axis=0)
            
        final_seg = np.zeros((3, ) + new_seg.shape)
        final_seg[0, :, :, :][np.where(new_seg != 0)] = 1
        final_seg[1, :, :, :][np.where((new_seg == 4) | (new_seg == 1))] = 1
        final_seg[2, :, :, :][np.where(new_seg == 4)] = 1
        
        return final_seg, start_depth, start_height, start_width
    

    def get_mask(self, path_seg):
        seg = sitk.GetArrayFromImage(sitk.ReadImage(path_seg))
        location = np.where(seg != 0)
       
        min_depth, max_depth = np.min(location[0]), np.max(location[0])
        min_height, max_height = np.min(location[1]), np.max(location[1])
        min_width, max_width = np.min(location[2]),  np.max(location[2])

        desired_depth = self.desired_depth
        desired_height = self.desired_height
        desired_width = self.desired_width

        new_volume = np.zeros((desired_depth, desired_height, desired_width))
        difference_depth = max_depth - min_depth
        difference_height = max_height - min_height
        difference_width = max_width - min_width

        if difference_depth < desired_depth:
            start_depth = np.random.randint(min_depth // 2, min_depth)
            end_depth = min(start_depth + desired_depth, 155)

            if end_depth == 155:
                start_depth = end_depth - desired_depth
        else:
            dice = np.random.uniform(0, 1)
            if dice > 0.5:
                start_depth = min_depth
                end_depth = start_depth + desired_depth
            else:
                end_depth = max_depth
                start_depth = max_depth - desired_depth

        if difference_height < desired_height:
            start_height = np.random.randint(min_height // 2, min_height)
            end_height = min(start_height + desired_height, 240)

            if end_height == 240:
                start_height = end_height - desired_height
        else:
            dice = np.random.uniform(0, 1)
            if dice > 0.5:
                start_height = min_height
                end_height = start_height + desired_height
            else:
                end_height = max_height
                start_height = max_height - desired_height

        if difference_width < desired_width:
            start_width = np.random.randint(min_width // 2, min_width)
            end_width = min(start_width + desired_width, 240)

            if end_width == 240:
                start_width = end_width - desired_width
        else:
            dice = np.random.uniform(0, 1)
            if dice > 0.5:
                start_width = min_width
                end_width = start_width + desired_width
            else:
                end_width = max_width
                start_width = max_width - desired_width
        new_seg = seg[start_depth: end_depth, start_height: end_height, start_width: end_width]
        final_seg = np.zeros((3, ) + new_seg.shape)
        final_seg[0, :, :, :][np.where(new_seg != 0)] = 1
        final_seg[1, :, :, :][np.where((new_seg == 4) | (new_seg == 1))] = 1
        final_seg[2, :, :, :][np.where(new_seg == 4)] = 1
        
        return final_seg, start_depth, start_height, start_width