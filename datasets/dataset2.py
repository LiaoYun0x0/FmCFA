import cv2
import os
from torch.utils.data import Dataset,DataLoader
import random
import numpy as np
import torch


        
class GLDataset(Dataset):
    def __init__(self,data_file,size=(320,320),stride=16):
        self.data_file = data_file
        with open(data_file, 'r') as f:
            self.train_data = f.readlines()

        self.size = size
        self.stride = stride # for generating gt-mask needed to compute local-feature loss
        self.query_pts = self._make_query_pts()
        self.mean = np.array([0.485, 0.456, 0.406],dtype=np.float32).reshape(3,1,1)
        self.std = np.array([0.229, 0.224, 0.225],dtype=np.float32).reshape(3,1,1)
        self.mean_sar = np.array([0.33247536, 0.33247536, 0.33247536],dtype=np.float32).reshape(3,1,1)
        self.std_sar = np.array([0.16769384, 0.16769384, 0.16769384],dtype=np.float32).reshape(3,1,1)
        self.mean_opt = np.array([0.31578836, 0.31578836, 0.31578836],dtype=np.float32).reshape(3,1,1)
        self.std_opt = np.array([0.1530546, 0.1530546 ,0.1530546],dtype=np.float32).reshape(3,1,1)


    def _read_file_paths(self,data_dir):
        assert os.path.isdir(data_dir), "%s should be a dir which contains images only"%data_dir
        file_paths = os.listdir(data_dir)
        return file_paths

    def __getitem__(self, index: int):
        n, opt, sar, x, y  = self.train_data[index].strip('\n').split(' ')
        opt_img_path = os.path.join(os.path.dirname(self.data_file), 'optical', opt)
        opt_img = cv2.imread(opt_img_path.replace('stage1_', ''))
        opt_img = cv2.cvtColor(opt_img,cv2.COLOR_BGR2RGB)
        x,y=int(x), int(y)
        opt_img = opt_img[y:y+512, x:x+512]
        #opt_img = opt_img[:256, :]
        #opt_img = cv2.copyMakeBorder(opt_img,256,0,0,0,cv2.BORDER_CONSTANT,value=(0,0,0))

        sar_img_path = os.path.join(os.path.dirname(self.data_file), 'sar', sar)
        sar_img = cv2.imread(sar_img_path.replace('stage1_', ''))
        sar_img = cv2.cvtColor(sar_img,cv2.COLOR_BGR2RGB)

        query = sar_img.transpose(2,0,1) 
        refer = opt_img.transpose(2,0,1)
        
        #query = ((query / 255.0) - self.mean) / self.std
        #refer = ((refer / 255.0) - self.mean) / self.std
        query = ((query / 255.0) - self.mean_sar) / self.std_sar
        refer = ((refer / 255.0) - self.mean_opt) / self.std_opt
        
        sample = {
            "refer":refer,
            "query":query,
            "pos": (int(x), int(y)),
            'class': n,
            'opt': opt,
            'sar': sar
            # "M": M,
            # "Mr": Mr,
            # "Mq": Mq
        }
        return sample
    
    def _generate_ref(self,ref, query, x, y, w, h):
        """
        通过sar和optical找到相对应的映射关系矩阵
        """

        refer,M = random_place(ref, query, x, y, w, h)
        # refer = ref_back.copy()
        # M = np.eye(3)

        #cv2.imshow("before:", query)
        query,Mq = self._aug_img(query) # 320x320x3, 3x3
        #cv2.imshow("after:", query)
        #cv2.waitKey()

        # np_img = np.ones_like(query) * 128
        # scale = 0.5
        # np_img[:int(self.size[1]*scale),:int(self.size[0]*scale)] = cv2.resize(query,None,fx=scale,fy=scale)
        # query = np_img
        refer,Mr = self._aug_img(refer)

            
        return query,refer, M, Mr,Mq
    
    def _generate_label(self,M,Mr,Mq,crop_coords=None,drop_mask=True):
        """
        M random_place
        Mr aug_refer
        Mq aug_query
        """
        x,y,w,h=crop_coords
        ncols, nrows = self.size[0] // self.stride, self.size[1] // self.stride
        label = np.zeros((ncols*nrows,ncols*nrows))
        Mq_inv = np.linalg.inv(Mq)
        src_pts = np.matmul(Mq_inv, self.query_pts.T) #self.query_pts (3x400) , shape:20x20x3, 变换位置
        mask0 = (0 <= src_pts[0,:]) & (src_pts[0,:] < self.size[0] ) & (0 <= src_pts[1,:]) & (src_pts[1,:]<self.size[1]) # 400x1

        dst_pts = np.matmul(M,src_pts)
        dst_pts = dst_pts / dst_pts[-1:,...]
        mask1 = (0 <= dst_pts[0,:]) & (dst_pts[0,:] < self.size[0]) & (0 <= dst_pts[1,:]) & (dst_pts[1,:]<self.size[1])

        refer_pts = np.matmul(Mr,dst_pts).T # (nrows*ncols, 3)
        mask2 = (0 <= refer_pts[:,0]) & (refer_pts[:,0]< self.size[0]) & (0 <= refer_pts[:,1]) & (refer_pts[:,1] < self.size[1])

        mask = drop_mask & mask0  & mask1 & mask2 
        
        match_index = np.int32(refer_pts[:,0]//self.stride + (refer_pts[:,1]//self.stride)*ncols)
        # num_match = np.sum(mask)
        # label[mask][np.arange(num_match),match_index[mask]] = 1
        indexes = np.arange(nrows*ncols)[mask]
        for index in indexes:
            label[index][match_index[index]] = 1
        return label
                
        
    def _make_query_pts(self):
        ncols, nrows = self.size[0] // self.stride, self.size[1] // self.stride
        half_stride = (self.stride-1) / 2
        xs = np.arange(ncols)
        ys = np.arange(nrows)
        xs = np.tile(xs[np.newaxis,:],(nrows,1))
        ys = np.tile(ys[:,np.newaxis],(1,ncols))
        ones = np.ones((nrows,ncols,1))
        grid = np.concatenate([xs[...,np.newaxis],ys[...,np.newaxis],ones],axis=-1)
        grid[:,:,:2] = grid[:,:,:2] * self.stride + half_stride  #(0:20, 0:20, 1) , shape:20x20x3
        return grid.reshape(-1,3) # (nrows*ncols , 3)
        
    def _random_flag(self,thresh=0.5):
        return np.random.rand(1) < thresh

    def _aug_img(self,img):
        h,w = img.shape[:2]
        matrix = np.eye(3)
        if self._random_flag(): 
            img = img[:,::-1,...].copy() # horizontal flip
            fM = np.array([
                [-1,0,w-1],
                [0,1,0],
                [0,0,1]
            ],np.float32)
            matrix = np.matmul(fM,matrix)
        
        if self._random_flag():
            img = img[::-1,:,...].copy() # vertical flip
            vfM = np.array([
                [1,0,0],
                [0,-1,h-1],
                [0,0,1]
            ],np.float32)
            matrix = np.matmul(vfM,matrix)

        if self._random_flag():
            img = change_lightness_contrast(img) # change light
        
        if self._random_flag():
            h,s,v = np.random.rand(3)/2.5 - 0.2
            img = random_distort_hsv(img,h,s,v)
        
        if self._random_flag():
            img = random_gauss_noise(img)
        
        if self._random_flag():
            img = random_mask(img)
        
        if self._random_flag():
            img,sh,sw = random_jitter(img,max_jitter=0.3)
            jM = np.array([
                [1,0,sw],
                [0,1,sh],
                [0,0,1]
            ],np.float32)
            matrix = np.matmul(jM,matrix)
            
        if self._random_flag(0):
            img,rM = random_rotation(img,max_degree=45)
            rM = np.concatenate([rM,np.array([[0,0,1]],np.float32)])
            matrix = np.matmul(rM,matrix)
        
        if self._random_flag():
            kernel = random.choice([3,5,7])
            img = blur_image(img,kernel)
        return img,matrix
            
    def __len__(self):
        return len(self.train_data)

def build_gl(
        train_data_file,
        test_data_file,
        size,
        stride):
    train_data = GLDataset(
        train_data_file,
        size=(320, 320),
        stride=8)
    test_data = GLDataset(
        test_data_file,
        size=(320, 320),
        stride=8)

    return train_data, test_data


