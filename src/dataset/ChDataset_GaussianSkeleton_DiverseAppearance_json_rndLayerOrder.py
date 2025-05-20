import os
import math
import json
import random
import numpy as np
from collections import OrderedDict
# import sys
# sys.path.insert(0, '/sensei-fs/users/diliu/zeqi_project/AnimateDiff/envs/lib/python3.10/site-packages')
import matplotlib as mpl
import torchvision.transforms as transforms
from omegaconf import OmegaConf
import torch
from PIL import Image
from torch.utils.data import Dataset
from einops import rearrange
import json
import cv2
import pandas as pd
import matplotlib.pylab as plt
import torchvision.transforms.functional as F
from matplotlib.patches import Circle, Rectangle, Ellipse
from transformers import CLIPImageProcessor



def set_all_seeds(seed):
    random.seed(seed)
    # os.environ('PYTHONHASHSEED') = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def draw_bodypose(canvas, candidate, subset, limbSeq):
    stickwidth = 4
    # limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
    #            [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
    #            [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    for i in range(len(subset)):
        index = int(subset[i])
        if index == -1:
            continue
        x, y = candidate[index][0:2]
        cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)
    for i in range(len(limbSeq)):
        index = subset[np.array(limbSeq[i]) ] #the output is a list, with index=limbSeq-1
        if -1 in index:
            continue
        cur_canvas = canvas.copy()
        Y = candidate[index.astype(int), 0]
        X = candidate[index.astype(int), 1]
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
        canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    # plt.imsave("preview.jpg", canvas[:, :, [2, 1, 0]])
    # plt.imshow(canvas[:, :, [2, 1, 0]])
    return canvas


class ChDataset(Dataset):
    def __init__(self, config=None):
        self.config = config
        image_anno=json.load(open(config["json_path"], 'r'), object_pairs_hook=OrderedDict)
        self.anno=image_anno["anno"]
        self.connected_joints=image_anno["connection"]
        if not type(self.config)==dict:
            self.config = OmegaConf.to_container(self.config)
        set_all_seeds(int(self.config["seed"]))
        self.max_resize_ratio=(1.5, 1.5)
        self.min_resize_ratio=(1., 1.)
        self.sample_size = np.array([self.config["sample_size"], self.config["sample_size"]])
        self.centercrop = transforms.CenterCrop(tuple(self.sample_size))
        self.limb_joints={
            'Left Leg': ['left hip', 'left knee', 'left heel', 'left toe'],
            'Right Leg': ['right hip', 'right knee', 'right heel', 'right toe'],
            'Left Arm': ['left shoulder', 'left elbow', 'left wrist', 'left hand'],
            'Right Arm': ['right shoulder', 'right elbow', 'right wrist', 'right hand'],
            }
        self.color_aug=[F.adjust_brightness, F.adjust_contrast, F.adjust_hue, F.adjust_saturation]
        self.composite_by_add = self.config.get("composite_by_add", False)
        # Below are only used if self.composite_by_add
        self.canonical_parts=self.config.get('canonical_parts', None)
        self.limb_parts = self.config.get('limb_parts', None)
        self.composite_order = self.config.get('part_order', None)
        self.draw_skeleton_order= self.config.get('node_order', None)
        self.canonical_part_alpha_folder = 'torso_canonical_part_rgba'
        self.arbitrary_part_alpha_folder = 'frames_part_alpha'
        self.background_dir=self.config.get("background_dir", [None, None])
        self.replace_path=self.config.get("replace_path", None)

        self.data = []
        ordered_target_joints=self.config["target_joints"]
        for i, (k,v) in enumerate(self.anno.items()):
            #k: image path, v: a dict of {joint_name: joint_coord}
            if self.replace_path is not None:
                k=k.replace(self.replace_path[0], self.replace_path[1])
                # canonical_path=canonical_path.replace(self.replace_path[0], self.replace_path[1])
            if self.background_dir[1]=='white':
                k=k.replace('512Frame', 'fg')
            if i ==0: #"x_range"
                image=Image.open(k)
                self.input_size=np.asarray(image.size) #need it to be a numpy array
                self.input_image_center=self.input_size/2
                self.resize_image_ratio=self.sample_size/self.input_size
                canonical_path=k 
                if self.background_dir[0]=='white':
                    canonical_path=canonical_path.replace('512Frame', 'fg')
            if self.background_dir[0]=='mask_comp':
                canonical_path=k.replace(k.split('/')[-2], 'mask_comp')

            joint_kp={}
            for joint_name, joint_coord in v.items():
                #joint_coord first is x second is y, so sample_size should actually be flipped.
                offset=joint_coord-self.input_image_center
                resized_joint_coord=self.resize_image_ratio*offset+self.sample_size/2
                joint_kp[joint_name]=resized_joint_coord
            print(joint_kp)

            if i ==0:
                canonical_joint_kp=joint_kp
            
            self.data.append({'image_path':k, \
                                'prompt':"",
                                # only this is ordered, so should control the order for all coordinates reading
                                "target_joints": ordered_target_joints, 
                                "joint_kp": joint_kp,
                                "canonical_joint_kp":canonical_joint_kp,
                                "canonical_image_path":canonical_path,
                                "composite_order": self.composite_order,
                                }) 
        print('data entries', len(self.data))

        if self.config.get('transform_norm', False)==False:
            self.transform = transforms.Compose([transforms.Resize(tuple(self.sample_size)),
                        transforms.ToTensor(),
                        # transforms.Lambda(lambda x: x* 2. - 1.)
                        ])
        else:
            print('self.transform has 0.5 normalization!')
            self.transform = transforms.Compose([transforms.Resize(tuple(self.sample_size)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5]),
                        # transforms.Lambda(lambda x: x* 2. - 1.)
                        ])

        self.pose_transform = transforms.Compose([transforms.Resize(tuple(self.sample_size)),
                        transforms.ToTensor(),
                        # transforms.Lambda(lambda x: x* 2. - 1.)
                        ])
        # self.is_image        = True
        # self.length = len(self.data)
        self.part_names=["Left Arm", "Right Arm","Left Leg","Right Leg"]
        self.texture=np.zeros((self.sample_size[0], self.sample_size[1], 3))
        for i in range(self.texture.shape[0]):
            for j in range(self.texture.shape[1]):
                self.texture[i,j]=np.asarray([i/(self.texture.shape[0]-1), j/(self.texture.shape[1]-1), 0])

        self.bone_interpolation=self.config.get("draw_joint_skeleton", "matplotlib")
        self.skeleton_layer_order = self.config.get("skeleton_layer_order", False) #not maintained for interpolation bone
        self.sample_n_frames = self.config.get("sample_n_frames", 0)
        self.target_bg_color=self.config.get("target_bg_color", 'white')
        self.skeleton_bg_color=self.config.get("skeleton_bg_color", 'white')

        self.clip_image_processor = CLIPImageProcessor()

    def __len__(self):
        return len(self.data) if self.config.get("dataset_size", None) is None else self.config["dataset_size"]

    def gen_drawing_info(self, idx):
        flip_coin=np.random.random()
        flip = 1
        if flip_coin < float(self.config.get("flip_prob", 0.5)):
            flip = -1#-1 is flip, 1 is no flip 
  
        resize_coin=np.random.random()
        resize_ratio=(1.,1.)
        if resize_coin < float(self.config.get("resize_prob", 0.)):
            resize_ratio=(np.random.uniform(self.min_resize_ratio[0], self.max_resize_ratio[0]),np.random.uniform(self.min_resize_ratio[1], self.max_resize_ratio[1]))
            
        if self.composite_by_add:
            change_composite_order_coin = np.random.random()
            composite_order = self.data[idx]['composite_order']
            if change_composite_order_coin < float(self.config.get("change_composite_order_prob", 0.5)):
                composite_order = np.random.permutation(composite_order)
        else:
            composite_order=None
        
        rotate_translate_coin=np.random.uniform()
        random_trans_y=np.random.randint(-self.sample_size[0]//4, high=self.sample_size[0]//4)
        random_trans_x=np.random.randint(-self.sample_size[1]//4, high=self.sample_size[1]//4)
        random_rotate_deg=np.random.uniform(self.config.get("random_rotate_deg", [-180, 180])[0], high=self.config.get("random_rotate_deg", [-180, 180])[1])
        return flip, resize_ratio, rotate_translate_coin, random_trans_y, random_trans_x, random_rotate_deg, \
                composite_order
    
    def __getitem__(self, idx):
        if len(self.data)==1:
            idx=0
        else:
            if self.config.get('mod', -1) != -1:
                print('idx, self.config[mod], len(self.data)', idx, self.config['mod'], len(self.data))
                idx=idx % self.config['mod']
            if self.config.get('fix_getitem_index', None):
                idx=(self.config['fix_getitem_index'])[idx]
        print('__getitem__ index from GaussianSkeleton_DiverseAppearance_json_rndLayerOrder', idx, self.data[idx]['image_path'])

        self.joint_bone_plot=OrderedDict()
        drawing_info = self.gen_drawing_info(idx)
        canonical_data=self.getonepair(idx, drawing_info, use_canonical_image=True)
        # self.joint_bone_plot=OrderedDict()
        data=self.getonepair(idx, drawing_info)

        ref_target_list, ref_vis_list=[],[] #ref_source_list=[]
        if int(self.config.get("n_pairs", 0))>0:
            ref_idx=np.random.choice(range(self.data[idx]["local_start"], self.data[idx]["local_end"]+1), size=int(self.config["n_pairs"]))
            for i in ref_idx:
                ref_data=self.getonepair(i, drawing_info)
                # ref_source_list.append(-ref_data["displacement"])
                ref_target_list.append(ref_data["pixel_values"])
                ref_vis_list.append(ref_data['visualize_target'])
        else:
            ref_idx=-1
            # ref_source_list.append(-torch.zeros((2, int(self.config["cond_H"]), int(self.config["cond_W"]))))
            ref_target_list.append(torch.zeros((3, self.sample_size[0], self.sample_size[1])))
            ref_vis_list.append(torch.zeros((3,self.sample_size[0], self.sample_size[1])))
        
        target=data["pixel_values"]
        target_vis=data['visualize_target']
        canonical_vis=canonical_data['visualize_target']
        canonical_target=canonical_data["pixel_values"]
        text=data["text"]
        ref_vis=torch.cat(ref_vis_list, 0)
        ref_target=torch.cat(ref_target_list, 0)
        background_img=torch.zeros_like(target)
        if self.background_dir[1]!=None and self.background_dir[1]!='white':
            background_img=Image.open(self.data[idx]['image_path'].replace('512Frame', self.background_dir[1])).resize(tuple(self.sample_size)).convert('RGB')
            background_img=self.transform(background_img)

        ref_img_pil = 255*(canonical_target.permute(1,2,0).numpy()+1)/2
        ref_img_pil = Image.fromarray(ref_img_pil.astype('uint8'))
        clip_image = self.clip_image_processor(
            images=ref_img_pil, return_tensors="pt"
        ).pixel_values[0]

        # n_targets controls how many target images to be generated for one get_item, so increases batch size
        # the total number of targets using the same graph will be n_targets*change_graph_freq
        if self.sample_n_frames > 0:
            more_target_list, more_vis_list=[],[]
            left, right = self.sample_n_frames//2, self.sample_n_frames - self.sample_n_frames//2
            left_bound, right_bound=self.data[idx]['local_start']+left, self.data[idx]['local_end']-right 
            left_empty=left_bound-idx
            right_empty=idx-right_bound
            left_start,right_start=idx-left+max(0, left_empty), idx+right-max(0, right_empty)
            if idx < left_bound:
                # print('left', self.data[idx]['local_idx'] , self.data[idx]['local_start'], left_bound, left_empty, left_start, right_bound, right_empty, right_start)
                more_target_list.extend([torch.zeros((3, self.sample_size[0], self.sample_size[1]))]*left_empty)
                more_vis_list.extend([torch.zeros((3,self.sample_size[0], self.sample_size[1]))]*left_empty)

            frame_idx=range(left_start, right_start)
            for i in frame_idx:
                more_target_data=self.getonepair(i, drawing_info)
                more_target_list.append(more_target_data["pixel_values"])
                more_vis_list.append(more_target_data['visualize_target'])
            if idx > right_bound:
                # print('right', self.data[idx]['local_idx'] , right_bound, right_empty)
                more_target_list.extend([torch.zeros((3, self.sample_size[0], self.sample_size[1]))]*right_empty)
                more_vis_list.extend([torch.zeros((3,self.sample_size[0], self.sample_size[1]))]*right_empty)
        
            target=torch.stack(more_target_list)
            target_vis=torch.stack(more_vis_list)
            canonical_vis=torch.stack([canonical_vis]*self.sample_n_frames)
            canonical_target=torch.stack([canonical_target]*self.sample_n_frames)
            ref_vis=torch.stack([ref_vis]*self.sample_n_frames)
            ref_target=torch.stack([ref_target]*self.sample_n_frames)
            background_img=torch.stack([background_img]*self.sample_n_frames)
            # print(len(more_target_list), target.size(), canonical_vis.size(), canonical_target.size(), target_vis.size(), ref_vis.size(), ref_target.size(), )
        data = {
            # 'ref_target':ref_target,
            # 'ref_idx': ref_idx,
            # 'ref_vis':ref_vis,
            'img': target,
            # 'txt': text,
            'bg_img':background_img,
            
            'tgt_pose': target_vis,
            'canonical_vis': canonical_vis,
            'ref_img': canonical_target,
            'clip_images': clip_image,
        }
        return data

    def composite_frame(self, idx, frame, composite_order, use_canonical_image):
        """
        use_canonical_image: where each part image should be read from, the canonical folder or frame 
                            indices folders.
        """
        if self.composite_by_add == False:
            frame = (np.clip(frame, 0, 1)*255).astype('uint8')
            return Image.fromarray(frame).convert('RGB')

        generate_dir=os.path.dirname(self.data[idx]['canonical_image_path'])
        frame_file = os.path.basename(self.data[idx]['image_path'])

        # generate a complete character figure by composing limb by limb
        # print('generate_dir, frame_file', generate_dir, frame_file)
        for i, name in enumerate(composite_order):
            # unchanged limbs.
            if not use_canonical_image:
                part_image = np.asarray(Image.open(os.path.join(generate_dir, self.arbitrary_part_alpha_folder, \
                                        name, frame_file)).resize(tuple(self.sample_size)))
                # print('not changed', os.path.join(generate_dir, self.arbitrary_part_alpha_folder, \
                                        # name, frame_file), color_mapping, color_mapping_coeff)
            else:
                part_image = np.asarray(Image.open(os.path.join(generate_dir, self.canonical_part_alpha_folder, \
                        name+'.png')).resize(tuple(self.sample_size)))
            part_image_alpha = part_image[:,:,-1:]
            part_image_color = part_image[:,:,:3]

            part_alpha = part_image_alpha.astype('float')/255
            part_color = np.asarray(part_image_color).astype('float')/255 #color_mapping()
            # print('part_alpha.shape', part_alpha.shape, part_color.shape, part_image_alpha.shape, part_image_color.shape)
            # if color_mapping is not None:
                # part_color = color_mapping[0]*part_color**2+color_mapping[1]*part_color+color_mapping[2]
            frame = (1-part_alpha)*frame[:,:,:3]+part_alpha*part_color

        return Image.fromarray(frame)


    def getonepair(self, idx, drawing_info, use_canonical_image=False, transform_canonical=False):
        # print('getonepair index from GaussianSkeleton_DiverseAppearance', idx, self.data[idx]['image_path'])
        flip, resize_ratio, rotate_translate_coin, random_trans_y, random_trans_x, random_rotate_deg, composite_info = drawing_info
        canonical_joint_kp = self.data[idx]['canonical_joint_kp']
        target_joints = self.data[idx]['target_joints']
        joint_kp = self.data[idx]['joint_kp']

        visualize_joint_kp=OrderedDict()
        for target_joint in target_joints:
            if use_canonical_image:
                x, y=canonical_joint_kp[target_joint]
            else:
                x, y=joint_kp[target_joint]
            offset_x, offset_y=resize_ratio[1]*(x-self.sample_size[1]/2), resize_ratio[0]*(y-self.sample_size[0]/2)
            new_x=self.sample_size[1]/2+flip*offset_x#-resize_ratio[1]*flip*center_x_offeset
            new_y=self.sample_size[0]/2+offset_y

            visualize_joint_kp[target_joint]=np.asarray([int(new_x), int(new_y)])

        if self.skeleton_bg_color=='black':
            skeleton_canvas=np.zeros((self.sample_size[0], self.sample_size[1], 3))
        elif self.skeleton_bg_color=='white':
            skeleton_canvas=np.ones((self.sample_size[0], self.sample_size[1], 3))
        
        if self.bone_interpolation == 'openpose':
            visualize_target=self.plot_joint_skeleton_openpose(visualize_joint_kp, skeleton_canvas)
        elif self.bone_interpolation == 'displacement':
            visualize_target=self.plot_skeleton(visualize_joint_kp, skeleton_canvas)
            # visualize_target.save(str(idx)+'.png')
        ############################
        if self.composite_by_add:
            if self.target_bg_color=='black':
                rgba = np.zeros((self.sample_size[0], self.sample_size[1], 4)) 
            elif self.target_bg_color=='white':
                # assume that even by add, only add limbs with the same skeleton
                rgba = np.ones((self.sample_size[0], self.sample_size[1], 4)) 
        else:
            if use_canonical_image:
                frame=Image.open(self.data[idx]['canonical_image_path']).convert('RGBA').resize(tuple(self.sample_size))
            else:
                frame=Image.open(self.data[idx]['image_path']).convert('RGBA').resize(tuple(self.sample_size))
            rgba = np.array(frame).astype('float')/255

            if self.target_bg_color=='black':
                #Note you can't do this if alpha is an input or output!!! It will cause color of the background blending
                # into the ground truth color, but in this dataset setting we are not predicting alpha so we are fine.
                rgba[:,:,:3] = rgba[:,:,:3]*rgba[:,:,-1:] + 0.*(1-rgba[:,:,-1:])
            elif self.target_bg_color=='white':
                rgba[:,:,:3] = rgba[:,:,:3]*rgba[:,:,-1:] + 1.*(1-rgba[:,:,-1:])

            # rgba[rgba[...,-1]==0] = [255,255,255,0]
        # frame returned has RGB 3 channels.
        frame = self.composite_frame(idx, rgba, composite_info, use_canonical_image)
        frame_resized=F.resize(frame, [int(resize_ratio[0]*self.sample_size[0]), int(resize_ratio[1]*self.sample_size[1])])
        frame_resized=self.centercrop(frame_resized)
        ############################
        if (not use_canonical_image) and (rotate_translate_coin<self.config.get("rotate_translate_prob", 0.)):
            if self.skeleton_bg_color=='black':
                visualize_target=F.affine(visualize_target, random_rotate_deg, [random_trans_x, random_trans_y], 1., 0., fill=0)
            elif self.skeleton_bg_color=='white':
                visualize_target=F.affine(visualize_target, random_rotate_deg, [random_trans_x, random_trans_y], 1., 0., fill=255)
            if self.target_bg_color=='black':
                frame_resized=F.affine(frame_resized, random_rotate_deg, [random_trans_x, random_trans_y], 1., 0., fill=0)
            elif self.target_bg_color=='white':
                frame_resized=F.affine(frame_resized, random_rotate_deg, [random_trans_x, random_trans_y], 1., 0., fill=255)

        if flip==-1:
            frame_resized=F.hflip(frame_resized)
            visualize_target=F.hflip(visualize_target)

        data = {
            'text': self.data[idx]["prompt"],
            'pixel_values': self.transform(frame_resized), 
            # 'displacement': displacement,
            'visualize_target': self.pose_transform(visualize_target),
        }
        return data

    def plot_joint_skeleton_displacement(self, transformed_images, frame):
        for i in range(len(transformed_images)):
            limb_color=transformed_images[i][:,:,:3]
            limb_alpha=transformed_images[i][:,:,-1:]
            frame=frame*(1-limb_alpha)+limb_color*limb_alpha
            
        frame = (np.clip(frame, 0, 1)*255).astype('uint8')
        return Image.fromarray(frame)

    def plot_joint_skeleton_openpose(self, joint_kp, frame):
        # print(self.connected_joints.keys(),joint_kp)
        size=max([int(parent) for parent in self.connected_joints if parent in joint_kp])
        size=1+max(size, max([int(parent) for parent in joint_kp]))
        subset=-np.ones((size, ))
        limbSeq=[]
        mapped_joint_kp=np.zeros((size, 2))
        for parent, children in self.connected_joints.items():
            if parent in joint_kp:
                openpose_index=int(parent)
                subset[openpose_index]=parent
                mapped_joint_kp[openpose_index][0]=joint_kp[parent][0]
                mapped_joint_kp[openpose_index][1]=joint_kp[parent][1]
                for c in children:
                    child_index=int(c)
                    subset[child_index]=c
                    limbSeq.append([openpose_index, child_index])
                    mapped_joint_kp[child_index][0]=joint_kp[c][0]
                    mapped_joint_kp[child_index][1]=joint_kp[c][1]

        frame = draw_bodypose(frame.astype('uint8'), mapped_joint_kp, subset, limbSeq)

        return Image.fromarray(frame)

    def plot_node_joint_bone(self, image, node_pos, node_parent_pos=None):
        x=node_pos
        if node_parent_pos is None:
            circle=cv2.circle(image, x.astype('int'), int(self.config.get("joint_std", 10)), (1,1,1), -1)
        else:
            y=node_parent_pos
            std=int(self.config.get("bone_std", 10))
            line_rect = 2*std
            k= 100000 if abs(y[0]-x[0])<0.01 else (y[1]-x[1])/(y[0]-x[0])
            rect=cv2.RotatedRect((x+y)/2, (line_rect, np.linalg.norm(x-y)), np.arctan(k)*180/np.pi-90)
            cv2.fillPoly(image, [rect.points().astype('int')], color=(1,1,1))

        return image

    def plot_skeleton(self, draw_target_joints, skeleton_canvas):
        visualize_joint_kp=draw_target_joints
        # if use_canonical_image:
        if len(self.joint_bone_plot)==0:
            #ordered dict so remember the sequence
            for target_joint in draw_target_joints:
                image=np.zeros((self.sample_size[0], self.sample_size[1], 3))
                image=self.plot_node_joint_bone(image, visualize_joint_kp[target_joint])
                self.joint_bone_plot[target_joint]=[image, visualize_joint_kp[target_joint], None, None, None]
            for parent, children in self.connected_joints.items():
                if parent in draw_target_joints:
                    x=visualize_joint_kp[parent]
                    #children order is fixed
                    for child in children:
                        if child in draw_target_joints:
                            y=visualize_joint_kp[child]
                            image, _, _, _, _=self.joint_bone_plot[child]
                            image=self.plot_node_joint_bone(image, visualize_joint_kp[child], node_parent_pos=x)
                            init_angle=np.arctan2([x[1]-y[1]],[x[0]-y[0]])[0]
                            self.joint_bone_plot[child]=[image, x, y, parent, init_angle]
            limb_num=len(draw_target_joints)
            intervals=np.linspace(0, 1, limb_num+2)[1:]
            equi_interval=intervals[1]-intervals[0]
            # the above only draws masks, below filles in with color
            # ordered, so i is the layer_order index
            for i, (k,v) in enumerate(self.joint_bone_plot.items()):
                if self.skeleton_layer_order:
                    depth_blue=intervals[i] #+np.random.uniform(-0.3*equi_interval, 0.3*equi_interval)
                    depth_blue=np.ones_like(self.texture[:,:,:1])*depth_blue
                    # v[0]=np.dstack((depth_blue, depth_blue, depth_blue, v[0][:,:,-2:-1]))
                    v[0]=np.dstack((self.texture[:,:,:2], depth_blue, v[0][:,:,-2:-1]))
                    # print(k, intervals[i])
                else:
                    v[0]=np.dstack((self.texture, v[0][:,:,-2:-1]))
            visualize_target=self.plot_joint_skeleton_displacement([v[0] for v in self.joint_bone_plot.values()], skeleton_canvas)
        else:
            joint_bone_plots=[]
            for target_joint, [joint_bone_plot, init_parent_pos, init_child_pos, parent_name, init_angle] in self.joint_bone_plot.items():
                if parent_name:
                    curr_parent_pos=visualize_joint_kp[parent_name]
                    curr_child_pos=visualize_joint_kp[target_joint]
                    tx, ty=curr_parent_pos-init_parent_pos
                    #arctan outputs rad
                    curr_angle=np.arctan2([curr_parent_pos[1]-curr_child_pos[1]],[curr_parent_pos[0]-curr_child_pos[0]])[0]
                    transform_angle=(curr_angle-init_angle)*180/np.pi
                    translation_matrix = np.array([
                                [1, 0, tx],
                                [0, 1, ty]
                            ], dtype=np.float32)
                    # print(node.name, node.init_draw_shape)
                    height, width = joint_bone_plot.shape[:2]
                    shape_image=np.pad(joint_bone_plot, ((height//2, height-height//2), (width//2, width-width//2), (0,0)), )
                    # print(height, width, shape_image.shape)
                    rotation_centor=np.copy(curr_parent_pos)
                    rotation_centor[0]+=width//2
                    rotation_centor[1]+=height//2
                    # input to cv2 width first height second
                    rotate_matrix = cv2.getRotationMatrix2D(center=(int(rotation_centor[0]), int(rotation_centor[1])),\
                                                            angle=-transform_angle, scale=1)
                    # must do rotation first and translation second!
                    translated_image = cv2.warpAffine(src=shape_image, M=translation_matrix, dsize=(2*width, 2*height))
                    init_dist=np.linalg.norm(init_parent_pos-init_child_pos)
                    curr_dist=np.linalg.norm(curr_parent_pos-curr_child_pos)
                    if abs(init_dist-curr_dist)>3:
                        # print(target_joint, parent_name, init_dist, curr_dist, abs(init_dist-curr_dist), 'changed length!')
                        restore_angle=init_angle*180/np.pi
                        restore_matrix = cv2.getRotationMatrix2D(center=(int(rotation_centor[0]), int(rotation_centor[1])),\
                                                            angle=restore_angle, scale=1)
                        translated_image = cv2.warpAffine(src=translated_image, M=restore_matrix, dsize=(2*width, 2*height))
                        translated_image = cv2.resize(translated_image,None,fx=curr_dist/init_dist,fy=1.,interpolation=cv2.INTER_NEAREST)
                        resized_w = translated_image.shape[1]
                        if resized_w>2*width:
                            displacement_rotation_center=int((curr_dist/init_dist-1)*rotation_centor[0])
                            translated_image=translated_image[:,displacement_rotation_center:displacement_rotation_center+2*width]
                        elif resized_w<2*width:
                            displacement_rotation_center=int((1-curr_dist/init_dist)*rotation_centor[0])
                            translated_image=np.pad(translated_image, ((0, 0), (displacement_rotation_center, 2*width-resized_w-displacement_rotation_center), (0,0)), )
                        restore_matrix = cv2.getRotationMatrix2D(center=(int(rotation_centor[0]), int(rotation_centor[1])),\
                                                            angle=-restore_angle, scale=1)
                        translated_image = cv2.warpAffine(src=translated_image, M=restore_matrix, dsize=(2*width, 2*height))
                        
                    rotated_image = cv2.warpAffine(src=translated_image, M=rotate_matrix, dsize=(2*width, 2*height))
                    image=rotated_image[height//2:height//2-height, width//2:width//2-width]
                else:
                    # image=joint_bone_plot
                    curr_parent_pos=visualize_joint_kp[target_joint]
                    tx, ty=curr_parent_pos-init_parent_pos
                    translation_matrix = np.array([
                                [1, 0, tx],
                                [0, 1, ty]
                            ], dtype=np.float32)
                    height, width = joint_bone_plot.shape[:2]
                    image = cv2.warpAffine(src=joint_bone_plot, M=translation_matrix, dsize=(width, height))
                joint_bone_plots.append(image)
            visualize_target=self.plot_joint_skeleton_displacement(joint_bone_plots, skeleton_canvas)
        return visualize_target
