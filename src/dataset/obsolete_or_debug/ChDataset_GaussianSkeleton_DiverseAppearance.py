import os
import math
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
from matplotlib.patches import Circle, Rectangle, Ellipse, Wedge, FancyArrow, FancyBboxPatch, BoxStyle
from matplotlib.lines import Line2D 
# from torchvision.io import ImageReadMode, read_image


def set_all_seeds(seed):
    random.seed(seed)
    # os.environ('PYTHONHASHSEED') = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class ChDataset(Dataset):
    def __init__(self, config=None):
        self.config = config or OmegaConf.create()
        set_all_seeds(int(self.config["seed"]))
        self.original_image_center=256
        self.max_resize_ratio=(1.5, 1.5)
        self.min_resize_ratio=(1., 1.)
        self.sample_size = (int(self.config["H"]), int(self.config["W"]))
        self.centercrop = transforms.CenterCrop(self.sample_size)
        self.resize_image_ratio=(int(self.config["H"])/512,int(self.config["W"])/512)
        self.encode_cond_ratio=(int(self.config["cond_H"])/512,int(self.config["cond_W"])/512)
        self.count = 0
        self.use_original_connection=self.config.get("use_original_connection", True)
        self.name_map={
            #only add Mouth when running RedMonster
            ('Head head', 'head', 'Head neck', 'neck', 'Head Pin head', 'Chloe', 'Footy','Head'): 'head',
            ('Left Pupil', 'Left Eye', 'Left_Eye', 'Left Eyebrow'): 'left eye',
            ('Right Pupil', 'Right Eye', 'Right_Eye', 'Right Eyebrow'): 'right eye',
            ('right shoulder', 'Right Arm'): 'right shoulder',
            ('left shoulder', 'Left Arm'): 'left shoulder',
            ('right elbow', 'Right Elbow', 'Relbow right elbow'): 'right elbow',
            ('left elbow', 'Left Elbow', 'Lelbow left elbow'): 'left elbow',
            ('left hip', 'Left Leg', 'Legs'): 'left hip',
            ('right hip', 'Right Leg', 'Legs'): 'right hip',
            ('left knee', 'Left Knee', 'Left knee'): 'left knee',
            ('right knee', 'Right Knee'): 'right knee',
            ('right wrist', 'Right Wrist', 'Right Hand right wrist', 'Rwrist right wrist'): 'right wrist',
            ('left wrist', 'Left Wrist', 'Left Hand left wrist', 'Lwrist left wrist'): 'left wrist',
            ('right heel', 'Right Heel'): 'right heel',
            ('left heel', 'Left Heel'): 'left heel',
            ('left toe', ): 'left toe',
            ('right toe', ): 'right toe',
            ('Left Hand', ): 'left hand', 
            ('Right Hand', ): 'right hand',
        }
        if self.config.get("include_theme_name", None) and ("Red Monster" in self.config["include_theme_name"]):
            print("Mouth added!")
            self.name_map[("Mounth")]='head'
        if self.use_original_connection:
            self.connected_joints={
                    'head': ['right shoulder', 'left shoulder', 'left hip', 'right hip', 'left eye', 'right eye'],
                    'left eye':[],
                    'right eye':[],
                    'right shoulder': ['right elbow',],
                    'left shoulder': ['left elbow',],
                    'right elbow': ['right wrist'],
                    'left elbow': ['left wrist'],
                    'left hip':['left knee'],
                    'right hip':['right knee'],
                    'left knee':['left heel'],
                    'right knee':['right heel'],
                    'right wrist':['right hand'],
                    'left wrist':['left hand'],
                    'right heel':['right toe'],
                    'left heel':['left toe'],
                }
        else:
            self.connected_joints={
                    'head': ['right shoulder', 'left shoulder', 'mid hip', 'mid eye'],
                    'mid eye': [],
                    'left eye':[],
                    'right eye':[],
                    'right shoulder': ['right elbow',],
                    'left shoulder': ['left elbow',],
                    'right elbow': ['right wrist'],
                    'left elbow': ['left wrist'],
                    'mid hip':['left hip','right hip'],
                    'left hip':['left knee'],
                    'right hip':['right knee'],
                    'left knee':['left heel'],
                    'right knee':['right heel'],
                    'right wrist':['right hand'],
                    'left wrist':['left hand'],
                    'right heel':['right toe'],
                    'left heel':['left toe'],

                }
        self.limb_joints={
            'Left Leg': ['left hip', 'left knee', 'left heel', 'left toe'],
            'Right Leg': ['right hip', 'right knee', 'right heel', 'right toe'],
            'Left Arm': ['left shoulder', 'left elbow', 'left wrist', 'left hand'],
            'Right Arm': ['right shoulder', 'right elbow', 'right wrist', 'right hand'],
            }
        # self.node_features = {k:np.random.uniform(size=(1,3)) for k in self.connected_joints}
        color=np.array([[0.90569166, 0.09516849, 0.07732852],
                [1.,  1., 0.0143458 ],
                [0.28073604, 0.09481623, 0.17072554],
                [0.9830016, 0.6706091, 0.06015595],
                [0.66497147, 0.79652931, 0.05386718],
                [0.0265808,  0.43040679, 0.87683378],
                [0.33099061, 0.27430081, 0.4627161 ],
                [0.74373924, 0.09893008, 0.8330483 ],
                [0.02916444, 0.21619655, 0.78014537],
                [0.29361416, 0.80542963, 0.06498297],
                [0.63614925, 0.88212252, 0.37375015],
                [0.96997283, 0.86286162, 0.85673081],
                [0.91456675, 0.78117677, 0.45258551],
                [0.96121327, 0.38110299, 0.55967198],
                [0.20762462, 0.55853708, 0.26328885],
                [0.01029422, 0.63126111, 0.50101218],
                [0.58197084, 0.25656854, 0.25199036],
                ])
        count=0
        self.node_features={}
        for k in self.connected_joints:
            self.node_features[k] = color[count:count+1]
            count+=1
        print('self.node_features', self.node_features)
        self.additional_colors = np.array([
                [0.56697362, 0.70319668, 0.76007258],
                [0.88752962, 0.33875029, 0.61470366],
                [0.74043657, 0.35687966, 0.41898697],
                [0.66387344, 0.19628544, 0.63938183],
                [0.44639799, 0.55315221, 0.30810827],
                # [0,0,0],
                [0,1,0],
                [0,1,1],
                [0.5, 0.5, 0.5],
                [0, 0.5, 0.5],
                [0.95853099, 0.5237316,  0.55298312],
                [0.73569738, 0.89817684, 0.12457296],
                [0.52289745, 0.02831192, 0.53875825],
                [0.5, 0.5, 0],
                [0.5, 0, 0.5],
                [0.5, 0.5, 1],
                [0.5, 1, 0.5],
                [1, 0.5, 0.5]
        ])
        self.original_facing_direction={
            'Animals': 'right',
            'Blobby': 'left',
            'Cecy': 'right',
            'Chibi': 'right',
            'Cowfolk': 'right',
            'Doppel': 'left',
            'Friday': 'right',
            'Hunter': 'right',
            'MegaRobot': 'left',
            'Nagisa': 'right',
            'Noodles': 'left',
            'Sci-Fi-Ranger': 'right',
            'The Professional': 'right',
            'Al': 'left',
            'Chloe': 'center',
            'Footy': 'center',
            'Hand': 'center',
            'Walkbot': 'left',
            'Sam': 'left',
            'Wampler': 'center',
            'Red Monster': 'center',
            'Lucy': 'right',
            'Pirate': 'right',
            'Hopscotch': 'right',
        }
        self.color_aug=[F.adjust_brightness, F.adjust_contrast, F.adjust_hue, F.adjust_saturation]
        self.canonical_parts=["Body Back", "Body", "Hair Back", "Head", "Body Front", "Hair Front"] 
        self.limb_parts = ["Left Leg", "Right Leg", "Left Arm", "Right Arm"]
        self.composite_order = {
            'left': ["Left Arm", "Left Leg", "Body Back", "Hair Back", "Body", "Head", "Body Front", "Hair Front", "Right Leg", "Right Arm"],
            'right':["Right Arm", "Right Leg", "Body Back", "Hair Back", "Body", "Head", "Body Front", "Hair Front", "Left Leg", "Left Arm"],
            'center':["Body Back", "Hair Back", "Body", "Head", "Body Front", "Hair Front", "Right Leg", "Left Leg", "Right Arm", "Left Arm"],
        }
        #TODO: toe and hand not updated for left and center 
        self.draw_skeleton_order={
            'left': ['left wrist', 'left elbow', 'left shoulder', 'left heel', 'left knee', 'left hip', \
                    'left eye', 'right eye', 'head',\
                    'right hip', 'right knee', 'right heel', 'right shoulder', 'right elbow', 'right wrist', ],
            'right': ['right wrist', 'right elbow', 'right shoulder', 'right heel', 'right knee', 'right hip', \
                    'left eye', 'right eye', 'head',\
                    'left hip', 'left knee', 'left heel', 'left shoulder', 'left elbow', 'left wrist', ],
            'center': ['head',  'left eye', 'right eye', \
                    'right hip', 'right knee', 'right heel', 'right toe', 'left hip', 'left knee', 'left heel', 'left toe',\
                    'right shoulder', 'right elbow', 'right wrist', 'right hand', 'left shoulder', 'left elbow', 'left wrist', 'left hand'],
        }
        self.canonical_part_alpha_folder = 'torso_canonical_part_rgba'
        self.arbitrary_part_alpha_folder = 'frames_part_alpha'
        #Produce an input frame by compositing each part in order. The opposite is by subtractions from a complete figure
        self.composite_by_add = self.config.get("composite_by_add", True)
        if not type(self.config)==dict:
            self.config = OmegaConf.to_container(self.config)

        self.data = []
        self.characters={}
        self.path_to_idx={}
        # index_filter=self.config.get("index_filter", [])
        # self.valid_index=[]
        data_dir = self.config["data_dir"]
        self.exclude_clip_names={}
        if 'exclude_clip_name' in self.config:
            for theme_clip in self.config['exclude_clip_name']:
                [theme, clip]=theme_clip.split(':')
                self.exclude_clip_names[theme]=clip.split(',')
        for character in os.listdir(self.config["data_dir"]):
            if 'ipynb' in character or 'DS_' in character:
                continue
            if 'include_theme_name' in self.config:
                if character not in self.config['include_theme_name']:
                    continue
            if 'exclude_theme_name' in self.config:
                if character in self.config['exclude_theme_name']:
                    continue
            self.data_dir = os.path.join(self.config["data_dir"], character)
            
            #TODO: Find the 1280 themes!!!
            if character in ['Footy']:
                r=1.
            # elif character in ['Lucy']:
            #     r=512/1280
            else:
                r=512/1080
            clips = [i for i in os.listdir(self.data_dir) if len(i)<3 and os.path.isdir(os.path.join(self.data_dir, i))]
            self.characters[character]=[]
            for clip in sorted(clips, key=lambda x: int(x)):
                if (character in self.exclude_clip_names) and (clip in self.exclude_clip_names[character]):
                    continue
                self.characters[character].append(clip)
                generate_dir=os.path.join(self.data_dir, clip)
                # coord in (x, y) order
                with open(os.path.join(generate_dir, 'canonical_keypoints.txt'), 'r') as read_f:
                    canonical_line=read_f.readlines()[0]
                canonical_joint_kp={}
                canonical_joints=canonical_line[:-3].split(', ')[1:]
                for joint in canonical_joints:
                    name, coord=joint.split(': ')
                    if name[0]==' ':
                        name=name[1:]                    
                    if (self.config.get("target_joints", None) is None) or \
                        ((self.config.get("target_joints", None) is not None) and (name in self.config["target_joints"])):

                        x, y = coord.split(',')

                        x=self.original_image_center+r*float(x)
                        y=self.original_image_center+r*float(y)
                        for k, v in self.name_map.items():
                            if (name in k):
                                mapped_name=v
                                canonical_joint_kp[mapped_name]=np.asarray([max(0, min(x, 511)), max(0, min(y, 511))])
                                break
                if self.config.get("target_joints", None) is not None:
                    target_joints=self.config["target_joints"]
                else:                
                    with open(os.path.join(generate_dir, 'target_joints.txt'), 'r') as f:
                        target_joints=[i.replace('\n', '') for i in f.readlines()]
                mapped_target_joints=[]
                for target_joint in target_joints:
                    for k, v in self.name_map.items():
                        if target_joint in k:
                            mapped_target_joints.append(v)
                ordered_target_joints=[]
                for name in self.draw_skeleton_order[self.original_facing_direction[character]]:
                    if name in mapped_target_joints:
                        ordered_target_joints.append(name)

                frames = [i for i in os.listdir(os.path.join(self.data_dir, clip, '512Frame')) if (('png' in i) and (i!='2186.png'))]
                local_start = self.count
                local_end = self.count+len(frames)-1
                print('len(frames)', len(frames), self.data_dir, clip)
                # coord in (x, y) order
                processed_csv = pd.read_csv(os.path.join(generate_dir, '512Frame', 'keypoints.csv'))
                for i in sorted(frames, key=lambda x: int(x.split('.')[0])):   
                    local_idx = self.count-local_start
                    joint_kp = processed_csv.iloc[local_idx]
                    mapped_joint_kp={}
                    for name in list(processed_csv.columns):
                        for k, v in self.name_map.items():
                            if (name in k):
                                x, y=joint_kp[name][1:-1].split(', ')
                                mapped_joint_kp[v]=np.asarray([float(x), float(y)])
                                break
                    if character in ['Lucy']:
                        mapped_joint_kp['right hip']=np.copy(mapped_joint_kp['left hip'])
                    # print(list(joint_kp), mapped_joint_kp.keys())
                    # if self.count in index_filter:              
                    self.data.append({'image_path':os.path.join(self.data_dir, clip, '512Frame', i), \
                                      'prompt':"",
                                      # only this is ordered, so should control the order for all coordinates reading
                                      "target_joints": ordered_target_joints, 
                                      "csv_joint_kp": mapped_joint_kp,
                                      "canonical":canonical_joint_kp,
                                      "local_start":local_start,
                                      "local_end":local_end,
                                      "local_idx": local_idx,
                                      "canonical_image_path":os.path.join(generate_dir, "canonical.png"),
                                      "composite_order": self.composite_order[self.original_facing_direction[character]],
                                      }) 
                    self.path_to_idx[os.path.join(self.data_dir, clip, '512Frame', i)]=self.count
                    self.count += 1
        print('data entries', len(self.data))

        self.transform = transforms.Compose([transforms.Resize(self.sample_size),
                    transforms.ToTensor(),
                    # transforms.Normalize([0.5], [0.5]),
                    # transforms.Lambda(lambda x: x* 2. - 1.)
                    ])
        self.is_image        = True
        self.length = len(self.data)
        self.Cartoon_Dataset_path=self.config.get("cartoon_dataset_dir", '')
        self.part_names=["Left Arm", "Right Arm","Left Leg","Right Leg"]
        self.bone_interpolation=self.config.get("draw_joint_skeleton", "matplotlib")
        if self.bone_interpolation=="displacement":
            self.texture=np.zeros((self.sample_size[0], self.sample_size[1], 3))
            for i in range(self.texture.shape[0]):
                for j in range(self.texture.shape[1]):
                    self.texture[i,j]=np.asarray([i/(self.texture.shape[0]-1), j/(self.texture.shape[1]-1), 0])
        self.appearance_guide = self.config.get("appearance_guide", "canonical") #support canonical, random
        self.skeleton_layer_order = self.config.get("skeleton_layer_order", False) #not maintained for interpolation bone
        self.sample_n_frames = self.config.get("sample_n_frames", 0)
        self.background_dir=self.config.get("background_dir", [None,None])
        self.target_bg_color=self.config.get("target_bg_color", 'white')
        self.skeleton_bg_color=self.config.get("skeleton_bg_color", 'white')

    def __len__(self):
        return len(self.data) if self.config.get("dataset_size", None) is None else self.config["dataset_size"]

    def gen_drawing_info(self, idx):
        background_img=None
        if self.background_dir is not None:
            Cartoon_Dataset_image_category = np.random.choice(os.listdir(self.background_dir))
            category_dir=os.path.join(self.background_dir, Cartoon_Dataset_image_category)
            background_file=random.choice([i for i in os.listdir(category_dir) if 'png' in i or 'jpg' in i])
            background_img=Image.open(os.path.join(category_dir, background_file)).resize(self.sample_size, 0).convert('RGB')
            background_img=np.asarray(background_img).astype('float')/255
        everywhere_cartoon_coin=np.random.random()
        everywhere_cartoon = False
        if everywhere_cartoon_coin < float(self.config.get("everywhere_cartoon_prob", 0.5)):
            everywhere_cartoon = True

        flip_coin=np.random.random()
        flip = 1
        if flip_coin < float(self.config.get("flip_prob", 0.5)):
            flip = -1#-1 is flip, 1 is no flip 
  
        resize_coin=np.random.random()
        resize_ratio=(1.,1.)
        if resize_coin < float(self.config.get("resize_prob", 0.)):
            resize_ratio=(np.random.uniform(self.min_resize_ratio[0], self.max_resize_ratio[0]),np.random.uniform(self.min_resize_ratio[1], self.max_resize_ratio[1]))
            
 
        change_composite_order_coin = np.random.random()
        composite_order = self.data[idx]['composite_order']
        if change_composite_order_coin < float(self.config.get("change_composite_order_prob", 0.5)):
            composite_order = np.random.permutation(composite_order)
        
        rotate_translate_coin=np.random.uniform()
        random_trans_y=np.random.randint(-self.sample_size[0]//4, high=self.sample_size[0]//4)
        random_trans_x=np.random.randint(-self.sample_size[1]//4, high=self.sample_size[1]//4)
        random_rotate_deg=np.random.uniform(self.config.get("random_rotate_deg", [-180, 180])[0], high=self.config.get("random_rotate_deg", [-180, 180])[1])
        return everywhere_cartoon, background_img, flip, resize_ratio, rotate_translate_coin, random_trans_y, random_trans_x, random_rotate_deg, \
                composite_order
    
    def __getitem__(self, idx):
        if len(self.data)==1:
            idx=0
        else:
            if self.config.get('mod', -1) != -1:
                idx=idx % self.config['mod']
            if self.config.get('fix_getitem_index', None):
                if len(self.config['fix_getitem_index'])==1:
                    idx=self.config['fix_getitem_index'][0]
                else:
                    idx=self.config['fix_getitem_index'][idx]
        # print('__getitem__ index from GaussianSkeleton_DiverseAppearance', idx)
        if self.bone_interpolation=='displacement':
            self.joint_bone_plot=OrderedDict()
        drawing_info = self.gen_drawing_info(idx)
        # if self.bone_interpolation==displacement, the first call will be used to fill self.joint_bone_plot
        if self.appearance_guide=='canonical':
            canonical_data=self.getonepair(idx, drawing_info, use_canonical_image=True)
        elif self.appearance_guide=='random':
            if self.config.get('appearance_guide_index',None):
                canonical_idx=self.config['appearance_guide_index']
            else:
                canonical_idx=np.random.choice(range(self.data[idx]["local_start"], self.data[idx]["local_end"]+1))
            canonical_data=self.getonepair(canonical_idx, drawing_info, no_transform=True)
        # everywhere_cartoon=drawing_info[0]
        # if everywhere_cartoon:
        #     data=self.getonepair(idx, drawing_info, use_canonical_image=True, transform_canonical=True)
        # else:
        data=self.getonepair(idx, drawing_info)

        ref_target_list, ref_vis_list=[],[] #ref_source_list=[]
        if int(self.config["n_pairs"])>0:
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
            # print(len(more_target_list), target.size(), canonical_vis.size(), canonical_target.size(), target_vis.size(), ref_vis.size(), ref_target.size(), )
        print(self.data[idx]["image_path"])
        data = {
            # 'ref_target':ref_target,
            # 'ref_idx': ref_idx,
            # 'ref_vis':ref_vis,
            'img': target,
            # 'txt': text,
            
            'tgt_pose': target_vis,
            # 'canonical_vis': canonical_vis,
            'ref_img': canonical_target,
        }
        self.count += 1
        return data

    def composite_frame(self, idx, frame, composite_order, use_canonical_image, joint_kp):
        """
        use_canonical_image: where each part image should be read from, the canonical folder or frame 
                            indices folders.
        """
        if self.composite_by_add == False:
            frame = (np.clip(frame, 0, 1)*255).astype('uint8')
            return Image.fromarray(frame).convert('RGB'), joint_kp, []

        generate_dir=os.path.dirname(self.data[idx]['canonical_image_path'])
        frame_file = os.path.basename(self.data[idx]['image_path'])

        # generate a complete character figure by composing limb by limb
        # print('generate_dir, frame_file', generate_dir, frame_file)
        for i, name in enumerate(composite_order):
            # unchanged limbs.
            if not use_canonical_image:
                part_image = np.asarray(Image.open(os.path.join(generate_dir, self.arbitrary_part_alpha_folder, \
                                        name, frame_file)).resize(self.sample_size))
                # print('not changed', os.path.join(generate_dir, self.arbitrary_part_alpha_folder, \
                                        # name, frame_file), color_mapping, color_mapping_coeff)
            else:
                part_image = np.asarray(Image.open(os.path.join(generate_dir, self.canonical_part_alpha_folder, \
                        name+'.png')).resize(self.sample_size))
            part_image_alpha = part_image[:,:,-1:]
            part_image_color = part_image[:,:,:3]

            part_alpha = part_image_alpha.astype('float')/255
            part_color = np.asarray(part_image_color).astype('float')/255 #color_mapping()
            # print('part_alpha.shape', part_alpha.shape, part_color.shape, part_image_alpha.shape, part_image_color.shape)
            # if color_mapping is not None:
                # part_color = color_mapping[0]*part_color**2+color_mapping[1]*part_color+color_mapping[2]
            frame = (1-part_alpha)*frame[:,:,:3]+part_alpha*part_color
        add_joints_kp=[]
        return Image.fromarray(frame), joint_kp, add_joints_kp


    def getonepair(self, idx, drawing_info, use_canonical_image=False, no_transform=False):
        # print('getonepair index from GaussianSkeleton_DiverseAppearance', idx)
        everywhere_cartoon, background_img, flip, resize_ratio, rotate_translate_coin, random_trans_y, random_trans_x, random_rotate_deg, composite_info = drawing_info
        canonical_joint_kp = self.data[idx]['canonical']
        target_joints = self.data[idx]['target_joints']
        joint_kp = self.data[idx]['csv_joint_kp']

        visualize_joint_kp={} #, visualize_canonical_joint_kp={}
        for target_joint in target_joints:
            if use_canonical_image:
                x, y=canonical_joint_kp[target_joint]
            else:
                x, y=joint_kp[target_joint]
            offset_x, offset_y=resize_ratio[1]*(x-self.original_image_center), resize_ratio[0]*(y-self.original_image_center)
            new_x=self.original_image_center+flip*offset_x#-resize_ratio[1]*flip*center_x_offeset
            new_y=self.original_image_center+offset_y

            # canonical_offset_x =resize_ratio[1]*(canonical_x-self.original_image_center)
            # canonical_offset_y = resize_ratio[0]*(canonical_y-self.original_image_center)
            # canonical_new_x=self.original_image_center+flip*canonical_offset_x#-resize_ratio[1]*flip*center_x_offeset
            # canonical_new_y=self.original_image_center+canonical_offset_y

            #Above are all processing on the original image size, below is scaling to the input size
            # canonical_encode_y=max(0, min(int(canonical_new_y*self.encode_cond_ratio[0]), int(self.config["cond_H"])-1))
            # canonical_encode_x=max(0, min(int(canonical_new_x*self.encode_cond_ratio[1]), int(self.config["cond_W"])-1))
            # displacement[0][canonical_encode_y][canonical_encode_x]=(y-canonical_new_y)/512
            # displacement[1][canonical_encode_y][canonical_encode_x]=(x-canonical_new_x)/512
            #Wanted to use *self.encode_cond_ratio[1], but now guess would be better to normalize within [0, 1]

            visualize_joint_kp[target_joint]=np.asarray([int(new_x*self.resize_image_ratio[1]), int(new_y*self.resize_image_ratio[0])])
            # visualize_canonical_joint_kp[target_joint]=(int(canonical_new_y*self.resize_image_ratio[0]), int(canonical_new_x*self.resize_image_ratio[1]))

        # rgba=np.array(frame_resized)
        # Make image transparent white anywhere it is transparent
        # rgba[rgba[...,-1]==0] = [255,255,255,0]

        # Make back into PIL Image and save
        # image=Image.fromarray(rgba).convert('RGB')
        ############################
        if "Walkbot" in self.data[idx]["image_path"]:        
            visualize_joint_kp['left eye']=visualize_joint_kp['right eye']
            visualize_joint_kp['right hip']=visualize_joint_kp['left hip']
        elif 'Red Monster' in self.data[idx]["image_path"]:  
            # print(visualize_joint_kp)
            # visualize_joint_kp['right shoulder']=np.asarray([254//2, 388//2])
            visualize_joint_kp['right shoulder']=np.asarray([174//2, 388//2])
            visualize_joint_kp['left shoulder']=np.asarray([334//2, 388//2])
            
        if not self.use_original_connection:
            draw_target_joints=target_joints.copy()
            if "Walkbot" in self.data[idx]["image_path"]:  
                draw_target_joints.append('left eye')
                draw_target_joints.append('right hip')
            eyes_existed, hips_existed=False,False
            if ('right eye' in visualize_joint_kp) and ('left eye' in visualize_joint_kp):
                visualize_joint_kp['mid eye']=np.asarray([(visualize_joint_kp['right eye'][0]+visualize_joint_kp['left eye'][0])//2,
                                                (visualize_joint_kp['right eye'][1]+visualize_joint_kp['left eye'][1])//2])
                draw_target_joints.remove('left eye')
                draw_target_joints.remove('right eye')
                eyes_existed=True
            if ('right hip' in visualize_joint_kp) and ('left hip' in visualize_joint_kp):
                visualize_joint_kp['mid hip']=np.asarray([(visualize_joint_kp['right hip'][0]+visualize_joint_kp['left hip'][0])//2,
                                            (visualize_joint_kp['right hip'][1]+visualize_joint_kp['left hip'][1])//2])
                hips_existed=True
            # mid eye and mid hip should be in the same column as head
            head_ind = draw_target_joints.index('head') if 'head' in draw_target_joints else 0
            if eyes_existed:
                draw_target_joints.insert(head_ind, 'mid eye')
            if hips_existed:
                draw_target_joints.insert(head_ind, 'mid hip')
            # draw_target_joints.remove('left heel')

        else:
            draw_target_joints=target_joints.copy()
            if 'Red Monster' in self.data[idx]["image_path"]:  
                draw_target_joints.append('right shoulder')
                draw_target_joints.append('left shoulder')

        if self.composite_by_add:
            if self.target_bg_color=='black':
                rgba = np.zeros((self.sample_size[0], self.sample_size[1], 4)) 
            elif self.target_bg_color=='white':
                # assume that even by add, only add limbs with the same skeleton
                rgba = np.ones((self.sample_size[0], self.sample_size[1], 4)) 
        else:
            if use_canonical_image:
                frame=Image.open(self.data[idx]['canonical_image_path']).convert('RGBA').resize(self.sample_size)
            else:
                frame=Image.open(self.data[idx]['image_path']).convert('RGBA').resize(self.sample_size)
            rgba = np.array(frame).astype('float')/255
            if background_img is None:
                if self.target_bg_color=='black':
                    #Note you can't do this if alpha is an input or output!!! It will cause color of the background blending
                    # into the ground truth color, but in this dataset setting we are not predicting alpha so we are fine.
                    rgba[:,:,:3] = rgba[:,:,:3]*rgba[:,:,-1:] + 0.*(1-rgba[:,:,-1:])
                elif self.target_bg_color=='white':
                    rgba[:,:,:3] = rgba[:,:,:3]*rgba[:,:,-1:] + 1.*(1-rgba[:,:,-1:])
            else:
                rgba[:,:,:3] = rgba[:,:,:3]*rgba[:,:,-1:] + background_img*(1-rgba[:,:,-1:])
            # rgba[rgba[...,-1]==0] = [255,255,255,0]
        # frame returned has RGB 3 channels.
        frame, visualize_joint_kp, additional_joints = self.composite_frame(idx, rgba, composite_info, use_canonical_image, visualize_joint_kp)
        # # since no alpha, calculate nonzero by pixel intensity.
        # alpha=np.linalg.norm(np.asarray(frame), axis=-1)
        # nonempty_y, nonempty_x=sorted(np.nonzero(alpha)[0]), sorted(np.nonzero(alpha)[1])
        # min_x, max_x, min_y, max_y=nonempty_x[0], nonempty_x[-1],nonempty_y[0],nonempty_y[-1]
        # height, width=max_y-min_y, max_x-min_x
        # h, w
        frame_resized=F.resize(frame, [int(resize_ratio[0]*self.sample_size[0]), int(resize_ratio[1]*self.sample_size[1])])
        frame_resized=self.centercrop(frame_resized)
        ############################
        if self.bone_interpolation == 'interpolation':
            visualize_target=self.plot_joint_skeleton_features(None, visualize_joint_kp, draw_target_joints)
        elif self.bone_interpolation == 'matplotlib':
            visualize_target=self.plot_joint_skeleton(None, visualize_joint_kp, target_joints, additional_joints)
        elif self.bone_interpolation == 'displacement':
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
                        for child in children:
                            if child in draw_target_joints:
                                y=visualize_joint_kp[child]
                                image, _, _, _, _=self.joint_bone_plot[child]
                                image=self.plot_node_joint_bone(image, visualize_joint_kp[child], node_parent_pos=x)
                                init_angle=np.arctan2([x[1]-y[1]],[x[0]-y[0]])[0]
                                self.joint_bone_plot[child]=[image, x, y, parent, init_angle]
                limb_num=len(draw_target_joints)
                intervals=np.linspace(0, 1, limb_num+1)[:-1]
                equi_interval=intervals[1]-intervals[0]
                # the above only draws masks, below filles in with color
                # ordered, so i is the layer_order index
                for i, (k,v) in enumerate(self.joint_bone_plot.items()):
                    if self.skeleton_layer_order:
                        depth_blue=intervals[limb_num-i-1]+np.random.uniform(-0.3*equi_interval, 0.3*equi_interval)
                        depth_blue=np.ones_like(self.texture[:,:,:1])*depth_blue
                        # v[0]=np.dstack((depth_blue, depth_blue, depth_blue, v[0][:,:,-2:-1]))
                        v[0]=np.dstack((self.texture[:,:,:2], depth_blue, v[0][:,:,-2:-1]))
                    else:
                        v[0]=np.dstack((self.texture, v[0][:,:,-2:-1]))
                if self.target_bg_color=='black':
                    visualize_target=self.plot_joint_skeleton_displacement([v[0] for v in self.joint_bone_plot.values()], \
                        np.zeros((self.sample_size[0], self.sample_size[1], 3)))
                elif self.target_bg_color=='white':
                    visualize_target=self.plot_joint_skeleton_displacement([v[0] for v in self.joint_bone_plot.values()], \
                        np.ones((self.sample_size[0], self.sample_size[1], 3)))
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
                if self.skeleton_bg_color=='black':
                    visualize_target=self.plot_joint_skeleton_displacement(joint_bone_plots, np.zeros((self.sample_size[0], self.sample_size[1], 3)))
                elif self.skeleton_bg_color=='white':
                    visualize_target=self.plot_joint_skeleton_displacement(joint_bone_plots, np.ones((self.sample_size[0], self.sample_size[1], 3)))

        
        if (not use_canonical_image) and (rotate_translate_coin<self.config.get("rotate_translate_prob", 0.)) and (not no_transform):
            visualize_target=F.affine(visualize_target, random_rotate_deg, [random_trans_x, random_trans_y], 1., 0., fill=255)
            frame_resized=F.affine(frame_resized, random_rotate_deg, [random_trans_x, random_trans_y], 1., 0., fill=0)

        if flip==-1:
            frame_resized=F.hflip(frame_resized)
            visualize_target=F.hflip(visualize_target)

        data = {
            'text': self.data[idx]["prompt"],
            'pixel_values': self.transform(frame_resized), 
            # 'displacement': displacement,
            'visualize_target': self.transform(visualize_target),
        }
        return data

    def plot_joint_skeleton_displacement(self, transformed_images, frame):
        for i in range(len(transformed_images)):
            limb_color=transformed_images[i][:,:,:3]
            limb_alpha=transformed_images[i][:,:,-1:]
            frame=frame*(1-limb_alpha)+limb_color*limb_alpha
            
        frame = (np.clip(frame, 0, 1)*255).astype('uint8')
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

    def plot_joint_skeleton(self, image, joint_kp, target_joints, additional_joints):
        if image is None:
            image=Image.fromarray((np.ones((self.sample_size[0], self.sample_size[1], 3))*255).astype('uint8'))
        # Create a figure. Equal aspect so circles look circular
        dpi = mpl.rcParams['figure.dpi']
        figsize=(2*256 / float(dpi), 2*256 / float(dpi))
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        # Show the image
        ax.imshow(image)
        for target_joint in target_joints:
            x, y =joint_kp[target_joint]
            for k, v in self.node_features.items():
                if target_joint in k:
                    color = v
                    break
            circ = Circle((x, y), 10, color=color)
            ax.add_patch(circ)
            
            for k, v in self.connected_joints.items():
                if target_joint in k:
#                     print(target_joint, k, joint_kp.keys())
                    for connected_joint in v:
                        if connected_joint in target_joints:
                            connect_x, connect_y=joint_kp[connected_joint]
                            line = Line2D([x, connect_x], [y, connect_y], color=color, linewidth=10.)
                            ax.add_line(line)
        
        count = 0
        for limb in additional_joints:
            for i in range(len(limb)):
                x, y = limb[i]
                circ = Circle((x, y), 10, color=self.additional_colors[count])
                ax.add_patch(circ)
                if i+1 < len(limb):
                    connect_x, connect_y = limb[i+1]
                    line = Line2D([x, connect_x], [y, connect_y], color=self.additional_colors[count], linewidth=10.)
                    ax.add_line(line)
                count += 1
            
        fig.canvas.draw()
        visualize_img=Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        plt.close()
        return visualize_img

    def plot_joint_skeleton_features(self, image, joint_kp, target_joints):
        # print(target_joints)
        # def find_k_feature(target_k):
        #     for k, v in self.connected_joints.items():
        #         if target_k in k:
        #             return self.node_features[k]
        if image is None:
            image=np.zeros((self.sample_size[0], self.sample_size[1], 3))

        for target_joint in target_joints:
            y, x=joint_kp[target_joint]
            for k, v in self.connected_joints.items():
                if target_joint in k:
                    target_color=self.node_features[k]
                    # self.gaussian_circle(x, y, image, target_color)
                    
                    for connected_joint in v:
                        if connected_joint in target_joints:
                            connect_y, connect_x=joint_kp[connected_joint]
                            connect_color=self.node_features[connected_joint]
                            self.interpolate_bone(np.array([x, y]), np.array([connect_x, connect_y]), target_color, connect_color, image)

        image = np.clip(image, 0, 1)
        visualize_img=Image.fromarray((image*255).astype('uint8'))
        return visualize_img


    def interpolate_bone(self, x, y, x_color, y_color, canvas, length_scale=1):
        dist_xy=np.linalg.norm(x-y)
        k= 100000 if abs(y[0]-x[0])<0.01 else (y[1]-x[1])/(y[0]-x[0])
        b=x[1]-k*x[0]
        # std=int(self.config.get("bone_std", 8))
        # line_rect = 4*std
        std=int(self.config.get("bone_std", 10))
        line_rect = 2*std
        image=np.zeros((self.sample_size[0], self.sample_size[1], 3))
        rect=cv2.RotatedRect((x+y)/2, (line_rect, length_scale*np.linalg.norm(x-y)), np.arctan(k)*180/np.pi-90)
        cv2.fillPoly(image, [rect.points().astype('int')], color=(255,255,255))
        for calc_y,calc_x in zip(*np.nonzero(image[:,:,0])):
            calc_point = np.array([calc_x,calc_y])
            closest_on_line=self.close_point_on_segment(calc_point, x, y)
            dist_calc_point_line=np.linalg.norm(calc_point-closest_on_line)
            dist_x=np.linalg.norm(closest_on_line-x)
            closest_on_line_color=(1-dist_x/dist_xy)*x_color+(dist_x/dist_xy)*y_color
            calc_point_color=np.exp(-0.5*((dist_calc_point_line/std)**2))*closest_on_line_color
            # canvas[calc_y][calc_x]=np.maximum(calc_point_color[0], canvas[calc_y][calc_x])
            canvas[calc_x][calc_y]=calc_point_color[0]

    def gaussian_circle(self, x, y, canvas, color):
        sigma = int(self.config.get("joint_std", 15))
        radius = 2 * sigma
        x_start=max(0, min(self.sample_size[1]-1, x-radius))
        x_end = max(0, min(self.sample_size[1]-1, x+radius))
        y_start=max(0, min(self.sample_size[0]-1, y-radius))
        y_end = max(0, min(self.sample_size[0]-1, y+radius))
        X = np.linspace(x_start, x_end, x_end-x_start+1)
        Y = np.linspace(y_start, y_end, y_end-y_start+1)
        X, Y = np.meshgrid(X, Y)
        Z = (np.exp(-((X-x)**2/(2*sigma**2) + (Y-y)**2/(2*sigma**2))))
        for xx,yy,zz in zip(X,Y,Z):
            # canvas[yy.astype('int'),xx.astype('int')]+= zz[:,None]*color
            canvas[yy.astype('int'),xx.astype('int')]=np.maximum(canvas[yy.astype('int'),xx.astype('int')], zz[:,None]*color)

    
    def close_point_on_segment(self, point, line_end_a, line_end_b):
        if np.linalg.norm(line_end_b-line_end_a)<1e-2:
            t=0
        else:
            t=np.dot(point-line_end_a, line_end_b-line_end_a)/np.dot(line_end_b-line_end_a, line_end_b-line_end_a)
            t=max(0, min(t, 1))
        closest=line_end_a+t*(line_end_b-line_end_a)
        return closest
