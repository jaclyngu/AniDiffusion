import os
import math
import random
import numpy as np
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
import bisect
# import opensimplex
from scipy.special import binom
from scipy import ndimage
import time
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

from animatediff.data.data_utils import *
from transformers import CLIPImageProcessor


def set_all_seeds(seed):
    random.seed(seed)
    # os.environ('PYTHONHASHSEED') = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class Node:
    def __init__(self,config,name,pos):
        self.config=config
        self.name=name
        self.index=int(self.name.split('x')[-1])
        self.dist2parent=-1
        self.parent=None
        self.children=[]
        self.angle2parent=-1 #in degree
        self.pos=pos #x first y second
        self.rotate_angle=[]
        self.sample_size = (int(self.config["H"]), int(self.config["W"]))
        self.move=False
        self.layer_order=-1 #the layer composition order, 0 is the closest to screen, 255 the background
        
        
    def gen_bezier_shape(self):
        bezier_additional_points_num=self.config.get("bezier_additional_points_num_range", [2, 4])
        bezier_aspect_ratio=self.config.get("bezier_aspect_ratio_range", [0.2, 0.4])
        bezier_scale=self.config.get("bezier_scale_range", [1, 1.8])
        rad=self.config.get("bezier_rad", 0.2)
        edgy=self.config.get("bezier_edgy", 0.05)

        self.fixed_points=np.array([self.pos, self.parent.pos])
        kp=np.array(self.fixed_points).T
        min_x, max_x=np.min(kp[0]),np.max(kp[0])
        min_y, max_y=np.min(kp[1]),np.max(kp[1])
        x_dist=max_x-min_x
        y_dist=max_y-min_y
        additional_points_num=np.random.randint(bezier_additional_points_num[0], bezier_additional_points_num[1]+1)
        if x_dist>y_dist:
            random_points_x=np.random.uniform(min_x, high=max_x, size=additional_points_num)
            max_y_ratio=np.random.uniform(bezier_aspect_ratio[0], bezier_aspect_ratio[1])
            random_points_y=np.random.uniform((min_y+max_y)/2-max_y_ratio*x_dist, high=(min_y+max_y)/2+max_y_ratio*x_dist, size=additional_points_num)
        else:     
            random_points_y=np.random.uniform(min_y, high=max_y, size=additional_points_num)
            max_x_ratio=np.random.uniform(bezier_aspect_ratio[0], bezier_aspect_ratio[1])
            random_points_x=np.random.uniform((min_x+max_x)/2-max_x_ratio*y_dist, high=(min_x+max_x)/2+max_x_ratio*y_dist, size=additional_points_num)

        random_points=np.hstack((random_points_x.reshape(-1, 1), random_points_y.reshape(-1, 1)))
        points=np.vstack((kp.T, random_points))
        self.x, self.y, _ = get_bezier_curve(points,rad=rad, edgy=edgy)
        self.x=np.clip(self.x, 0, self.sample_size[0])
        self.y=np.clip(self.y, 0, self.sample_size[1])

        
    def init_draw_shape(self, texture_file):
        self.gen_bezier_shape()
        dpi = mpl.rcParams['figure.dpi']
        figsize=(self.sample_size[0]/ float(dpi), self.sample_size[1] / float(dpi))
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        # this needs to be np.ones not np.zeros and wouldn't influce the final composition.
        blank=(np.ones((self.sample_size[0], self.sample_size[1], 3))*255).astype('uint8')
        ax.imshow(blank)
        
        plt.xlim([0, self.sample_size[1]])
        plt.ylim([self.sample_size[0], 0])
        plt.plot(self.x, self.y, color='r')

        fig.canvas.draw()
        visualize_img=Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        plt.close()

        np_image=np.array(visualize_img)
        mask_contour=255-np_image[:,:,2]
        # mask_contour[mask_contour>0]=255

        filled_circle=ndimage.binary_fill_holes(mask_contour).astype(int)
        # texture_file=texture_sources[i]
        # print('texture_file.shape', texture_file.shape,texture_file.max(), texture_file.min())
        limb_image=np.concatenate((texture_file, filled_circle[:,:,None]),axis=-1)
        #range [0, 1] with some overflows, RGBA
        self.init_pos_shape=limb_image

class ChDataset(Dataset):
    def __init__(self, config=None):
        self.config = config or OmegaConf.create()
        set_all_seeds(int(self.config["seed"]))
        self.original_image_center=256
        self.sample_size = (int(self.config["H"]), int(self.config["W"]))
        self.centercrop = transforms.CenterCrop(self.sample_size)
        self.resize_image_ratio=(int(self.config["H"])/512,int(self.config["W"])/512)
        self.encode_cond_ratio=(int(self.config["cond_H"])/512,int(self.config["cond_W"])/512)
        self.n_pairs=self.config.get("n_pairs", 0)
        self.n_targets=self.config.get("n_targets", 0)
        self.count = 0

        # self.node_features = {k:np.random.uniform(size=(1,3)) for k in self.connected_joints}
        self.color=np.array([[0.90569166, 0.09516849, 0.07732852],
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
                [0.20762462, 0.55853708, 0.26328885]])
        self.additional_colors = np.array([
                [0.56697362, 0.70319668, 0.76007258],
                [0.88752962, 0.33875029, 0.61470366],
                [0.74043657, 0.35687966, 0.41898697],
                [0.66387344, 0.19628544, 0.63938183],
                [0.44639799, 0.55315221, 0.30810827],
                [0.01029422, 0.63126111, 0.50101218],
                [0.58197084, 0.25656854, 0.25199036],
                # [0,0,0],
                [0,1,0],
                [0,1,1],
                [0.95853099, 0.5237316,  0.55298312],
                [0.73569738, 0.89817684, 0.12457296],
                [0.52289745, 0.02831192, 0.53875825],
                [0.5, 0.5, 0.5],
                [0, 0.5, 0.5],
                [0.5, 0.5, 0],
                [0.5, 0, 0.5],
                [0.5, 0.5, 1],
                [0.5, 1, 0.5],
                [1, 0.5, 0.5]
        ])
        self.available_colors=np.concatenate((self.color, self.additional_colors))
        self.Cartoon_Dataset_path=self.config.get("cartoon_dataset_dir", '')
        self.FractalNoise_Dataset_path=self.config.get("fractal_noise_dataset_dir", '')
        self.transform = transforms.Compose([transforms.Resize(self.sample_size),
            transforms.ToTensor(),
            # transforms.Normalize([0.5], [0.5]),
            # transforms.Lambda(lambda x: x* 2. - 1.)
            ])
            
        self.change_graph_freq=self.config.get("change_graph_freq", 1)
        self.max_move_limb=self.config.get("max_move_limbs", 20)
        self.limb_rotate_angle_range=self.config.get("limb_rotate_angle_range", [-60, 60])
        self.limb_rotate_mode=self.config.get("limb_rotate_mode", "random") #[random, linspace]
        self.skeleton_layer_order = self.config.get("skeleton_layer_order", False) #not maintained for interpolation bone
        self.bone_interpolation=self.config.get("draw_joint_skeleton", "matplotlib") #"matplotlib", interpolation, displacement
        if self.bone_interpolation=="displacement":
            self.texture=np.zeros((self.sample_size[0], self.sample_size[1], 3))
            for i in range(self.texture.shape[0]):
                for j in range(self.texture.shape[1]):
                    self.texture[i,j]=np.asarray([i/(self.texture.shape[0]-1), j/(self.texture.shape[1]-1), 0])
        self.data_aug_prob=self.config.get("data_aug_prob", 0)
        self.appearance_guide = self.config.get("appearance_guide", "canonical") #support canonical, random
        self.scale_leaves_skeleton= self.config.get("scale_leaves_skeleton", None)
        self.background_dir=self.config.get("background_dir", None)
        self.target_bg_color=self.config.get("target_bg_color", 'white')
        self.skeleton_bg_color=self.config.get("skeleton_bg_color", 'white')

        self.clip_image_processor = CLIPImageProcessor()
    
    def __len__(self):
        return 10000

    def random_aug(self, img_np):
        """
        img_np: uint8 numpy array
        """
        elastic_kwargs={
            'grid': 10, 
            'distortion_range': [0.2, 1], 
            'each_grid_prob': 0.85
        }
        uniform_darken_kwargs={
            'darken_range':[-50, 50]
        }
        blur_kwargs= {'sigma_range':[0.5, 2], 'kernel': 5}

        elastic_prob = 0.5
        uniform_darken_prob = 0.5
        blur_prob = 0.5
        coins = np.random.uniform(size=3)
        if coins[0]<elastic_prob:
            # print('elastic')
            img_np = elastic_deform(img_np, **elastic_kwargs)
        if coins[1]<uniform_darken_prob:
            # print('niform_darken')
            img_np = darken_uniform(img_np, **uniform_darken_kwargs)
        if coins[2]<blur_prob:
            # print('blur')
            img_np = add_blur(img_np, **blur_kwargs)
        return img_np


    def gen_drawing_info(self):
        self.background_img=None
        if self.background_dir is not None:
            Cartoon_Dataset_image_category = np.random.choice(os.listdir(self.background_dir))
            category_dir=os.path.join(self.background_dir, Cartoon_Dataset_image_category)
            background_file=random.choice([i for i in os.listdir(category_dir) if 'png' in i or 'jpg' in i])
            background_img=Image.open(os.path.join(category_dir, background_file)).resize(self.sample_size, 0).convert('RGB')
            self.background_img=np.asarray(background_img).astype('float')/255

        limb_num_max = int(self.config.get("limb_num", 20))
        limb_num=np.random.randint(2, limb_num_max+1)

        different_texture_per_limb=self.config.get("different_texture_per_limb", False)
        available_texture_sources=self.config.get("available_texture_sources", ['Fractal_Noise'])
        texture_probs = self.config.get("texture_probs", np.linspace(0,1,len(available_texture_sources)+1)[1:])
        data_aug_coin=np.random.random()

        if not different_texture_per_limb:
            texture_source_coin=np.random.random()
            texture_source_idx=bisect.bisect_left(texture_probs, texture_source_coin)
            texture_source=available_texture_sources[texture_source_idx]
            if texture_source=='Cartoon_Dataset':
                Cartoon_Dataset_image_category = np.random.choice(os.listdir(self.Cartoon_Dataset_path))
                Cartoon_Dataset_image_path = np.random.choice(os.listdir(os.path.join(\
                                        self.Cartoon_Dataset_path, Cartoon_Dataset_image_category)))
                Cartoon_Dataset_image_path = os.path.join(self.Cartoon_Dataset_path, Cartoon_Dataset_image_category, Cartoon_Dataset_image_path)      
                texture_file=np.asarray(Image.open(Cartoon_Dataset_image_path).resize(self.sample_size))
            elif texture_source=='Fractal_Noise':
                Fractal_Noise_image_path = np.random.choice([i for i in os.listdir(self.FractalNoise_Dataset_path) if 'png' in i])
                texture_file=np.asarray(Image.open(os.path.join(self.FractalNoise_Dataset_path,Fractal_Noise_image_path)).resize(self.sample_size))
                # shape=np.array([self.sample_size[0], self.sample_size[1],3])
                # res = (shape/shape.max())*5
                # start=time.time()
                # pix = createFractalNoise(shape.tolist(), 7, seed=np.random.randint(99999999999))
                # print('plot fractal noise', time.time()-start)
                # texture_file = pix*0.7+0.5
            else:
                raise Exception(s, 'texture source not implemented!')
            if data_aug_coin<self.data_aug_prob:
                texture_file=self.random_aug(texture_file)
            texture_sources=[texture_file.astype('float')/255]*limb_num
        else:
            texture_sources=[]
            for i in range(limb_num):
                texture_source_coin=np.random.random()
                texture_source_idx=bisect.bisect_left(texture_probs, texture_source_coin)
                texture_source=available_texture_sources[texture_source_idx]
                if texture_source=='Cartoon_Dataset':
                    Cartoon_Dataset_image_category = np.random.choice(os.listdir(self.Cartoon_Dataset_path))
                    Cartoon_Dataset_image_path = np.random.choice(os.listdir(os.path.join(\
                                            self.Cartoon_Dataset_path, Cartoon_Dataset_image_category)))
                    Cartoon_Dataset_image_path = os.path.join(self.Cartoon_Dataset_path, Cartoon_Dataset_image_category, Cartoon_Dataset_image_path)      
                    texture_file=np.asarray(Image.open(Cartoon_Dataset_image_path).resize(self.sample_size))
                elif texture_source=='Fractal_Noise':
                    Fractal_Noise_image_path = np.random.choice([i for i in os.listdir(self.FractalNoise_Dataset_path) if 'png' in i])
                    texture_file=np.asarray(Image.open(os.path.join(self.FractalNoise_Dataset_path,Fractal_Noise_image_path)).resize(self.sample_size))
                else:
                    raise Exception(s, 'texture source not implemented!')
                if data_aug_coin<self.data_aug_prob:
                    texture_file=self.random_aug(texture_file)
                texture_sources.append(texture_file.astype('float')/255)


        available_colors=np.random.permutation(self.available_colors)
        return texture_sources, available_colors, limb_num

    def perturb_canonical(self, root):
        """
        Random rotation in the bfs order to make the plot no longer forced-directed canonical
        """   
        frontier=[root]
        visited=[]
        while frontier:
            visit_node=frontier.pop(0)
            visited.append(visit_node)
            for c in visit_node.children:
                if c not in visited:
                    c.init_parent_pos=visit_node.pos
                    transform_angle=np.random.randint(-45, 45)
                    curr_angle=c.angle2parent+transform_angle
                    c.angle2parent=curr_angle
                    curr_angle=curr_angle*np.pi/180
                    # print('node.pos', node.pos, node.dist2parent, curr_angle, node.angle2parent, node.angle2parent+transform_angle)
                    c.pos=np.array([c.parent.pos[0]+c.dist2parent*np.cos(curr_angle),\
                                c.parent.pos[1]+c.dist2parent*np.sin(curr_angle)])
                    frontier.append(c)


    def assign_bfs_depth(self, root):
        frontier=[root]
        visited=[]
        order_idx=0
        while frontier:
            visit_node=frontier.pop(0)
            visit_node.layer_order=order_idx
            visited.append(visit_node)
            order_idx+=1
            for c in visit_node.children:
                if c not in visited:
                    frontier.append(c)


    def update_graph(self):
        texture_sources, self.curr_available_colors, limb_num = self.gen_drawing_info()
        ###############generate network######################
        target_joints=['x0']
        pairs=[]
        for i in range(1, limb_num+1):
            anchor='x0'#np.random.choice(target_joints)
            x='x'+str(i)
            target_joints.append(x)
            pairs.append([anchor, x])

        G = nx.DiGraph()
        G.add_nodes_from(target_joints)
        for i,j in pairs:
            #random the length
            G.add_edge(i,j, len=np.random.uniform(1,high=2.5))
        pos =graphviz_layout(G, prog="neato")
        # print(np.stack(list(pos.values())).max(0),np.stack(list(pos.values())).min(0))
        x_max,y_max=np.stack(list(pos.values())).max(0)
        x_min,y_min=np.stack(list(pos.values())).min(0)
        # a way to set the original plot size (h=w) to make sure all shapes are completely contained.
        h=100*(math.ceil(max(x_max,y_max)/100))+100
        x_mid,y_mid=(x_max+x_min)/2, (y_max+y_min)/2
        x_dis,y_dis=h/2-x_mid,h/2-y_mid
        # Draw graph
        pos_np={k:np.array([np.clip([v[0]+x_dis],0,h)[0]*self.sample_size[0]/h,\
                            np.clip([v[1]+y_dis],0,h)[0]*self.sample_size[1]/h]) for k,v in pos.items()}
        #####################################################
        A=nx.adjacency_matrix(G)
        adj=np.array(A.todense())
        self.nodes=[]
        for name in G.nodes:
            self.nodes.append(Node(self.config, name, pos_np[name]))
        self.non_leaves=[]
        for i in range(len(adj)):
            if adj[i].sum()>0:
                self.non_leaves.append(self.nodes[i])
            for j in range(len(adj[i])):
                if adj[i][j]==1:
                    self.nodes[i].children.append(self.nodes[j])
                    self.nodes[j].parent=self.nodes[i]
                    self.nodes[j].init_parent_pos=self.nodes[i].pos
                    self.nodes[j].dist2parent=np.linalg.norm(pos_np[self.nodes[j].name]-pos_np[self.nodes[i].name])
                    self.nodes[j].angle2parent=np.arctan2([self.nodes[j].pos[1]-self.nodes[i].pos[1]],[self.nodes[j].pos[0]-self.nodes[i].pos[0]])[0]
                    # all stored in degrees
                    self.nodes[j].angle2parent=self.nodes[j].angle2parent*180/np.pi 
                    if self.limb_rotate_mode=='random': 
                        self.nodes[j].rotate_angle=np.random.uniform(self.limb_rotate_angle_range[0], self.limb_rotate_angle_range[1], size=1+self.n_pairs+self.n_targets)
                    elif self.limb_rotate_mode=='linspace': 
                        self.nodes[j].rotate_angle=np.linspace(self.limb_rotate_angle_range[0], self.limb_rotate_angle_range[1], 2+self.n_pairs+self.n_targets)

        if self.appearance_guide =='random':
            self.perturb_canonical(self.nodes[0])

        count=0
        intervals=None
        if self.skeleton_layer_order:
            self.assign_bfs_depth(self.nodes[0])
            intervals=np.array([0.25, 0.5, 0.75])#np.linspace(0, 1, limb_num+1)[:-1]
        # print(limb_num, intervals)
        for node in self.nodes:
            if node.parent:
                node.init_draw_shape(texture_sources[count])
                count += 1
            if self.bone_interpolation=="displacement":
                self.plot_node_joint_bone(node, limb_num, intervals)


    def plot_node_joint_bone(self, node, limb_num, intervals):
        image=np.zeros((self.sample_size[0], self.sample_size[1], 3))
        x=node.pos

        if (node.parent) and (node not in self.non_leaves) and (self.scale_leaves_skeleton): #is a leaf, randomly scale bone
            y=node.parent.pos
            x=np.random.uniform(self.scale_leaves_skeleton[0], high=self.scale_leaves_skeleton[1])*(x-y)+y
            # print(x, node.pos) not the same, value got copied in memeory
        circle=cv2.circle(image, x.astype('int'), int(self.config.get("joint_std", 10)), (1,1,1), -1)
        if node.parent:
            y=node.parent.pos
            k= 100000 if abs(y[0]-x[0])<0.01 else (y[1]-x[1])/(y[0]-x[0])
            std=int(self.config.get("bone_std", 10))
            line_rect = 2*std
            rect=cv2.RotatedRect((x+y)/2, (line_rect, np.linalg.norm(x-y)), np.arctan(k)*180/np.pi-90)
            cv2.fillPoly(image, [rect.points().astype('int')], color=(1,1,1))
        # node.joint_bone_plot is RGBA where the shape is only controlled by A. RGB are all image color.
        if intervals is not None:
            equi_interval=intervals[1]-intervals[0]
            # print(limb_num, node.layer_order,len(intervals))
            depth_blue=intervals[limb_num-node.layer_order]#+np.random.uniform(-0.3*equi_interval, 0.3*equi_interval)
            # print(node.layer_order, depth_blue)
            depth_blue=np.ones_like(self.texture[:,:,:1])*depth_blue
            # node.joint_bone_plot=np.dstack((depth_blue, depth_blue, depth_blue, image[:,:,-2:-1]))
            node.joint_bone_plot=np.dstack((self.texture[:,:,:2], depth_blue, image[:,:,-2:-1]))
        else:
            node.joint_bone_plot=np.dstack((self.texture, image[:,:,-2:-1]))
        

    def __getitem__(self, idx):
        if self.count % self.change_graph_freq==0:
            self.update_graph()
        else:
            for i in self.nodes:
                i.move=False
                i.rotate_angle=np.random.uniform(self.limb_rotate_angle_range[0], self.limb_rotate_angle_range[1], size=1+self.n_pairs+self.n_targets)
                if i.name=='x2':
                    i.rotate_angle*=0
        move_starts=[self.nodes[1], self.nodes[2]]
        # move_all_limbs_coin=np.random.random()
        # if move_all_limbs_coin<self.config.get('move_all_limbs_prob', 0.3):
        #     move_starts=self.nodes[0]
        # else:
        #     move_starts=np.random.choice(self.nodes[1:], size=np.random.randint(1, high=1+min(len(self.nodes[1:]), \
        #                                                 self.max_move_limb)), replace=False)
        if (not isinstance(move_starts, list)) and (not isinstance(move_starts, np.ndarray)):
            move_starts=[move_starts]
        # print([i.name for i in  move_starts])
        for move_start in move_starts:
            move_start.move=True
        
        canonical_data=self.getonepair(-1)
        data=self.getonepair(0)
        
        ref_target_list, ref_vis_list=[],[]
        # n_pairs controls how many appearance reference images to be concat, so increases the channel
        if self.n_pairs>0:
            ref_idx=range(1,self.n_pairs+1)
            for i in ref_idx:
                ref_data=self.getonepair(i)
                ref_target_list.append(ref_data["pixel_values"])
                ref_vis_list.append(ref_data['visualize_target'])
        else:
            ref_idx=-1
            ref_target_list.append(torch.zeros((3, self.sample_size[0], self.sample_size[1])))
            ref_vis_list.append(torch.zeros((3,self.sample_size[0], self.sample_size[1])))

        target=data["pixel_values"]
        target_vis=data['visualize_target']
        canonical_vis=canonical_data['visualize_target']
        canonical_target=canonical_data["pixel_values"]
        text=data["text"]
        ref_vis=torch.cat(ref_vis_list, 0)
        ref_target=torch.cat(ref_target_list, 0)

        ref_img_pil = 255*(canonical_target.permute(1,2,0).numpy()+1)/2
        ref_img_pil = Image.fromarray(ref_img_pil.astype('uint8'))
        clip_image = self.clip_image_processor(
            images=ref_img_pil, return_tensors="pt"
        ).pixel_values[0]

        # n_targets controls how many target images to be generated for one get_item, so increases batch size
        # the total number of targets using the same graph will be n_targets*change_graph_freq
        if self.n_targets>0:
            more_target_list, more_vis_list=[],[]
            ref_idx=range(self.n_pairs+1, self.n_targets+self.n_pairs+1)
            for i in ref_idx:
                more_target_data=self.getonepair(i)
                more_target_list.append(more_target_data["pixel_values"])
                more_vis_list.append(more_target_data['visualize_target'])
        
            target=torch.stack([target]+more_target_list)
            target_vis=torch.stack([target_vis]+more_vis_list)
            canonical_vis=torch.stack([canonical_vis]*(self.n_targets+1))
            canonical_target=torch.stack([canonical_target]*(self.n_targets+1))
            ref_vis=torch.stack([ref_vis]*(self.n_targets+1))
            ref_target=torch.stack([ref_target]*(self.n_targets+1))
            clip_image=torch.stack([clip_image]*(self.n_targets+1))
        
        data = {
            # 'ref_target':ref_target,
            # 'ref_idx': ref_idx,
            # 'ref_vis':ref_vis,
            'img': target,
            # 'txt': text,
            
            'tgt_pose': target_vis,
            # 'canonical_vis': canonical_vis,
            'ref_img': canonical_target,
            'clip_images': clip_image,
        }
        self.count += 1
        return data

    def bfs(self, root, transform_idx):
        frontier=[root]
        visited=[]
        transformed_images, transformed_joint_bones = [], []
        #transformed_joint_bones only used if 
        if self.bone_interpolation=="displacement":
            transformed_joint_bones=[frontier[0].joint_bone_plot] # the root's untransformed joint circle
        while frontier:
            visit_node=frontier.pop(0)
            visited.append(visit_node)
            #no need to transform the root
            if visit_node != root:
                if visit_node.move:
                    rotated_image, rotated_joint_bone=self.transform_one_limb(visit_node, transform_idx)
                else:
                    rotated_image, rotated_joint_bone=self.transform_one_limb(visit_node, -1)
                transformed_images.append(rotated_image)
                transformed_joint_bones.append(rotated_joint_bone)
            for c in visit_node.children:
                if c not in visited:
                    # but don't inherit from parent if parent.move == False
                    if visit_node.move==True:
                        c.move=True
                    frontier.append(c)
        return visited, transformed_images, transformed_joint_bones

    def transform_one_limb(self, node, transform_idx):
        if transform_idx==-1:
            transform_angle=0
        else:
            if node.name=='x2':
                transform_angle=0
            else:
                transform_angle=node.rotate_angle[transform_idx]
        curr_angle=node.angle2parent+transform_angle
        # print(node.name, curr_angle)
        curr_angle=curr_angle*np.pi/180
        # print('node.pos', node.pos, node.dist2parent, curr_angle, node.angle2parent, node.angle2parent+transform_angle)
        node.pos=np.array([node.parent.pos[0]+node.dist2parent*np.cos(curr_angle),\
                    node.parent.pos[1]+node.dist2parent*np.sin(curr_angle)])
        
        tx, ty=node.parent.pos-node.init_parent_pos
        # print(node.name, node.parent.name, tx, ty, node.pos, node.init_parent_pos, node.parent.pos,transform_angle,np.cos(curr_angle))
        translation_matrix = np.array([
                    [1, 0, tx],
                    [0, 1, ty]
                ], dtype=np.float32)
        # print(node.name, node.init_draw_shape)
        height, width = node.init_pos_shape.shape[:2]
        shape_image=np.pad(node.init_pos_shape, ((height//2, height-height//2), (width//2, width-width//2), (0,0)), )
        # print(height, width, shape_image.shape)
        rotation_centor=np.copy(node.parent.pos)
        rotation_centor[0]+=width//2
        rotation_centor[1]+=height//2
        # input to cv2 width first height second
        rotate_matrix = cv2.getRotationMatrix2D(center=rotation_centor, angle=-transform_angle, scale=1)
        # must do rotation first and translation second!
        translated_image = cv2.warpAffine(src=shape_image, M=translation_matrix, dsize=(2*width, 2*height))
        rotated_image = cv2.warpAffine(src=translated_image, M=rotate_matrix, dsize=(2*width, 2*height)) 
        rotated_image = rotated_image[height//2:height//2-height, width//2:width//2-width]

        if self.bone_interpolation=="displacement":
            #the skeleton is also drawn here
            joint_bone= np.pad(node.joint_bone_plot, ((height//2, height-height//2), (width//2, width-width//2), (0,0)), )
            translated_joint_bone = cv2.warpAffine(src=joint_bone, M=translation_matrix, dsize=(2*width, 2*height))
            rotated_joint_bone = cv2.warpAffine(src=translated_joint_bone, M=rotate_matrix, dsize=(2*width, 2*height))
            rotated_joint_bone = rotated_joint_bone[height//2:height//2-height, width//2:width//2-width]
        else:
            rotated_joint_bone = None
        return rotated_image, rotated_joint_bone
        
    def composite_frame(self, transformed_images, frame):
        for i in range(len(transformed_images)):
            limb_color=transformed_images[i][:,:,:3]
            limb_alpha=transformed_images[i][:,:,-1:]
            frame=frame*(1-limb_alpha)+limb_color*limb_alpha
            
        frame = (np.clip(frame, 0, 1)*255).astype('uint8')
        return Image.fromarray(frame)


    def getonepair(self, idx):
        # assume that even by add, only add limbs with the same skeleton
        if self.background_img is None:
            if self.target_bg_color=='black':
                rgba = np.zeros((self.sample_size[0], self.sample_size[1], 3)) 
            elif self.target_bg_color=='white':
                rgba = np.ones((self.sample_size[0], self.sample_size[1], 3)) 
        else:
            rgba=np.copy(self.background_img)
        transformed_ordered_nodes, transformed_images, transformed_joint_bones =self.bfs(self.nodes[0], idx//2)
        if idx %2==0:
            frame = self.composite_frame(transformed_images, rgba)
        elif idx %2==1:
            frame = self.composite_frame(transformed_images[::-1], rgba)

        if self.skeleton_bg_color=='black':
            rgba_skeleton=np.zeros((self.sample_size[0], self.sample_size[1], 3)) 
        elif self.skeleton_bg_color=='white':
            rgba_skeleton=np.ones((self.sample_size[0], self.sample_size[1], 3)) 

        if self.bone_interpolation=='interpolation':
            visualize_target= self.plot_joint_skeleton_features(None, transformed_ordered_nodes)
        elif self.bone_interpolation=='matplotlib':
            visualize_target=self.plot_joint_skeleton(None, transformed_ordered_nodes)
        elif self.bone_interpolation=='displacement':
            if idx %2==0:
                visualize_target=self.composite_frame(transformed_joint_bones, rgba_skeleton)
            elif idx %2==1:
                first_depth_blue=transformed_joint_bones[1][:,:,3].copy()
                transformed_joint_bones[1][:,:,3]=transformed_joint_bones[2][:,:,3].copy()
                transformed_joint_bones[2][:,:,3]=first_depth_blue
                visualize_target=self.composite_frame(transformed_joint_bones[::-1], rgba_skeleton)

        rotate_translate_coin=np.random.uniform()
        if (idx!=-1) and (rotate_translate_coin<self.config.get("rotate_translate_prob", 0.)):
            random_trans_y=np.random.randint(-self.sample_size[0]//4, high=self.sample_size[0]//4)
            random_trans_x=np.random.randint(-self.sample_size[1]//4, high=self.sample_size[1]//4)
            random_rotate_deg=np.random.uniform(-90, high=90)
            if self.skeleton_bg_color=='black':
                visualize_target=F.affine(visualize_target, random_rotate_deg, [random_trans_x, random_trans_y], 1., 0., fill=0)
            elif self.skeleton_bg_color=='white':
                visualize_target=F.affine(visualize_target, random_rotate_deg, [random_trans_x, random_trans_y], 1., 0., fill=255)
            if self.target_bg_color=='black':
                frame=F.affine(frame, random_rotate_deg, [random_trans_x, random_trans_y], 1., 0., fill=0)
            elif self.target_bg_color=='white':
                frame=F.affine(frame, random_rotate_deg, [random_trans_x, random_trans_y], 1., 0., fill=255)

        data = {
            'text': '',
            'pixel_values': self.transform(frame), 
            'visualize_target': self.transform(visualize_target),
        }
        return data

    def plot_joint_skeleton(self, image, transformed_ordered_nodes):
        if image is None:
            image=Image.fromarray((np.ones((self.sample_size[0], self.sample_size[1], 3))*255).astype('uint8'))
        # Create a figure. Equal aspect so circles look circular
        dpi = mpl.rcParams['figure.dpi']
        figsize=(2*256 / float(dpi), 2*256 / float(dpi))
        # figsize=(self.sample_size[0] / float(dpi), self.sample_size[1] / float(dpi))
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])
        plt.xlim([0,self.sample_size[1]])
        plt.ylim([self.sample_size[0],0])
        ax.axis('off')
        # Show the image
        ax.imshow(image)
        count = 0
        for node in transformed_ordered_nodes:
            # print('skeletton parent', parent)
            x,y=node.pos
            circ = Circle((x,y), 10, color=self.curr_available_colors[count:count+1])
            ax.add_patch(circ)

            if node.parent is not None:
                line = Line2D([node.parent.pos[0], x], [node.parent.pos[1], y], color=self.curr_available_colors[count:count+1], linewidth=10.)
                ax.add_line(line)
            count += 1

        fig.canvas.draw()
        visualize_img=Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        plt.close()
        return visualize_img


    def plot_joint_skeleton_features(self, image, transformed_ordered_nodes):   
        if image is None:
            # image=Image.fromarray((np.ones((self.sample_size[0], self.sample_size[1], 3))*255).astype('uint8'))
            image=np.zeros((self.sample_size[0], self.sample_size[1], 3))
        if self.skeleton_layer_order:
            order_layer=-1*np.ones(self.sample_size[0], self.sample_size[1], 3)
        else:
            order_layer=None
        # Create a figure. Equal aspect so circles look circular
        dpi = mpl.rcParams['figure.dpi']
        figsize=(2*256 / float(dpi), 2*256 / float(dpi))
        # figsize=(self.sample_size[0] / float(dpi), self.sample_size[1] / float(dpi))
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])
        plt.xlim([0,self.sample_size[1]])
        plt.ylim([self.sample_size[0],0])
        ax.axis('off')
        # Show the image
        ax.imshow(image)
        count = 0
        for node in transformed_ordered_nodes:
            node.color=self.curr_available_colors[count:count+1]
            # self.gaussian_circle(node, image, order_layer)
            if node.parent is not None:
                self.interpolate_bone(node, node.parent, image, order_layer)
            count += 1
        # for node in transformed_ordered_nodes:
        image = np.clip(image, 0, 1)
        visualize_img=Image.fromarray((image*255).astype('uint8'))
        plt.close()
        return visualize_img


    def interpolate_bone(self, node, connected_node, canvas, order_layer):
        x = node.pos
        y=connected_node.pos
        x_color=node.color 
        y_color= connected_node.color
        dist_xy=np.linalg.norm(x-y)
        k= 100000 if abs(y[0]-x[0])<0.01 else (y[1]-x[1])/(y[0]-x[0])
        b=x[1]-k*x[0]
        std=int(self.config.get("bone_std", 10))
        line_rect = 2*std
        image=np.zeros((self.sample_size[0], self.sample_size[1], 3))
        rect=cv2.RotatedRect((x+y)/2, (line_rect, max(0, np.linalg.norm(x-y))), np.arctan(k)*180/np.pi-90)
        cv2.fillPoly(image, [rect.points().astype('int')], color=(255,255,255))
        for calc_y,calc_x in zip(*np.nonzero(image[:,:,0])):
            if order_layer:
                order_layer[calc_y][calc_x]=node.layer_order
            calc_point = np.array([calc_x,calc_y])
            closest_on_line=self.close_point_on_segment(calc_point, x, y)
            dist_calc_point_line=np.linalg.norm(calc_point-closest_on_line)
            dist_x=np.linalg.norm(closest_on_line-x)
            closest_on_line_color=(1-dist_x/dist_xy)*x_color+(dist_x/dist_xy)*y_color
            calc_point_color=np.exp(-0.5*((dist_calc_point_line/std)**2))*closest_on_line_color
            # canvas[calc_y][calc_x] = np.maximum(calc_point_color[0], canvas[calc_y][calc_x]) 
            canvas[calc_y][calc_x] = calc_point_color[0]


    def gaussian_circle(self, node, canvas, order_layer):
        x,y=node.pos
        sigma = int(self.config.get("joint_std", 15))
        radius = 2 * sigma
        x_start=int(max(0, min(self.sample_size[1]-1, x-radius)))
        x_end = int(max(0, min(self.sample_size[1]-1, x+radius)))
        y_start=int(max(0, min(self.sample_size[0]-1, y-radius)))
        y_end = int(max(0, min(self.sample_size[0]-1, y+radius)))
        X = np.linspace(x_start, x_end, x_end-x_start+1)
        Y = np.linspace(y_start, y_end, y_end-y_start+1)
        X, Y = np.meshgrid(X, Y)
        Z = (np.exp(-((X-x)**2/(2*sigma**2) + (Y-y)**2/(2*sigma**2))))
        for xx,yy,zz in zip(X,Y,Z):
            #xx [10. 11. 12. ] y [56. 56. 56.] z [0.49126265 0.5250165 0.55860132]
            # canvas[yy.astype('int'),xx.astype('int')]+= zz[:,None]*color
            if order_layer:
                order_layer[yy.astype('int'),xx.astype('int')]=node.layer_order
            # canvas[yy.astype('int'),xx.astype('int')]=np.maximum(canvas[yy.astype('int'),xx.astype('int')], zz[:,None]*node.color)
            canvas[yy.astype('int'),xx.astype('int')]=zz[:,None]*node.color
            # for i in range(len(xx)):
            #     if ((xx[i]-x)**2+(yy[i]-y)**2)**0.5<=radius:
            #         canvas[yy[i].astype('int'),xx[i].astype('int')]=zz[i]*node.color

    
    def close_point_on_segment(self, point, line_end_a, line_end_b):
        t=np.dot(point-line_end_a, line_end_b-line_end_a)/np.dot(line_end_b-line_end_a, line_end_b-line_end_a)
        t=max(0, min(t, 1))
        closest=line_end_a+t*(line_end_b-line_end_a)
        return closest

