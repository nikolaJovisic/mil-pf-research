import torch
import torch.nn as nn
import gmic
import cv2
import numpy as np
import read_config_file
from torchvision import transforms
import torch.nn.functional as F

device = 'cuda' 

class SILmodel(nn.Module):
    def __init__(self, config_params):
        super(SILmodel, self).__init__()
        self.activation = config_params['activation']
        self.featureextractormodel = config_params['femodel']
        self.extra = config_params['extra']
        self.topkpatch = config_params['topkpatch']
        self.pretrained = config_params['pretrained']
        self.channel = config_params['channel']
        self.regionpooling = config_params['regionpooling']
        self.learningtype = config_params['learningtype']
        
        if self.featureextractormodel:
            self.feature_extractor = gmic._gmic(config_params['gmic_parameters'])
    
    def forward(self, x, eval_mode=True):
        y_local, y_global, y_fusion, saliency_map, patch_locations, patches, patch_attns, h_crops, concat_vec = self.feature_extractor(x, eval_mode)
        return y_local, y_global, y_fusion, saliency_map, patch_locations, patches, patch_attns, h_crops, concat_vec


class GMICModel(nn.Module):
    def __init__(self, config_params, model_path):
        super(GMICModel, self).__init__()
        self.model = SILmodel(config_params)
        self.transforms = self.preprocess()
        self.load_model(model_path)
    
    def forward(self, images):
        images = self.transforms(images)
        images = images.unsqueeze(0).to(device)
        # import pudb; pudb.set_trace()
        y_local, y_global, y_fusion, saliency_map, patch_locations, patches, patch_attns, h_crops, concat_vec = self.model(images)
        return concat_vec

    def load_model(self, model_path):
        checkpoint = torch.load(path_to_model)
        print("checkpoint epoch and loss:", checkpoint['epoch'], checkpoint['loss'])  
        self.model.load_state_dict(checkpoint['state_dict'])

    def preprocess(self):
        mean = [0.485, 0.456, 0.406]
        std_dev = [0.229, 0.224, 0.225]
        preprocess_test = transforms.Compose([transforms.Resize((2944,1920)), transforms.Normalize(mean=mean, std=std_dev)])
        return preprocess_test    

def build_model(config_params, model_path, device):
    model = GMICModel(config_params, model_path)
    print('model loaded')
    model.to(device)
    return model

def load_image(img_path):
    img = cv2.imread(img_path,-1)
    img_dtype = img.dtype
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB).astype(np.float32)
    if img_dtype=='uint16':
        img/=65535
    img = torch.from_numpy(img.transpose((2, 0, 1))).contiguous()
    return img


# GMIC original
import imageio
def load_image_gmic(image_path, view):
    """
    Loads a png or hdf5 image as floats and flips according to its view.
    """
    image = np.array(imageio.imread(image_path))
    image = image.astype(np.float32)
    if view == 'R':
        image = np.fliplr(image)
    return image

class GMICOrigModel(nn.Module):
    def __init__(self, model_path):
        super(GMICOrigModel, self).__init__()
        
        parameters = {
                "device_type":device,
                "cam_size": (46, 30),
                "K": 6,
                "crop_shape": (256, 256),
                "percent_t":0.02,
                "post_processing_dim": 256,
                "num_classes": 2,
                "learningtype": "SIL"
        }
        
        self.model = gmic.GMIC(parameters)
        self.load_model(model_path)
    
    def forward(self, image):
        image = self.preprocess(image)
        print(image.shape)
        # import pudb; pudb.set_trace()
        return self.model(image)
        y_local, y_global, y_fusion, saliency_map, patch_locations, patches, patch_attns, h_crops, concat_vec = self.model(image)
        return concat_vec

    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location="cuda")
        self.model.load_state_dict(checkpoint, strict=False)


    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: torch.Tensor of shape (H, W), (C, H, W), or (B, C, H, W)
        returns: torch.Tensor of shape (B, C, 2944, 1920)
        """
        # Handle input shape
        images = images[:, 0, ...]
        if images.ndim == 2:
            images = images.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif images.ndim == 3:
            if images.shape[0] in (1, 3):
                images = images.unsqueeze(0)  # (1, C, H, W)
            else:
                images = images.unsqueeze(1)  # assume (B, H, W)
        elif images.ndim != 4:
            raise ValueError(f"Unsupported tensor shape {images.shape}")

        # Resize to (2944, 1920)
        images = F.interpolate(images, size=(2944, 1920), mode='bicubic', align_corners=False)

        # Per-image normalization (zero mean, unit variance)
        mean = images.mean(dim=(2, 3), keepdim=True)
        std = images.std(dim=(2, 3), keepdim=True).clamp_min(1e-5)
        images = (images - mean) / std

        return images


    
def build_model(device):
    model_path = '/lustre/GMIC/models/sample_model_1.p'
    model = GMICOrigModel(model_path)
    model.to(device)
    model.eval()
    return model



if __name__=='__main__':
    #read arguments
#     config_file = '/lustre/pathak/runs/embed/train_gmic_embed/config_80_8.ini'    
#     config_params = read_config_file.read_config_file(config_file)

#     path_to_model = '/lustre/pathak/runs/embed/train_gmic_embed/model_80_8.tar' 
        
#     model = build_model(config_params, path_to_model, device)
#     model.eval()

    img_path = '/lustre/pathak/data/embed/processed_images/1003266064259604/LCC_png_images-cohort_1-842ee092b0e93ac76ed7ce56f797a023d4bdbdda58b4e6bfe36d08f2-9ab611e24fb7f13a3996adf5705c57194db4124cafa5eb485df6a256-a00cc4664f2ddbce5d7623ac4c16159f7ab5aef768b56fb13b8dd140.png'
#     img = load_image(img_path)
#     print(img.shape)
#     pred = model(img)
#     print(pred.shape)
    
    
    # GMIC
    model_path = '/lustre/GMIC/models/sample_model_1.p'
    model = GMICOrigModel(model_path)
    model.to(device)
    model.eval()
    print('model loaded')
    print(model)
    
    img = load_image_gmic(img_path, view='L')
    pred = model(img)
    print(pred.shape)
    
    
    # """Pathak Testing code"""
    # total_images=0
    # correct = 0
    # s=0
    # batch_test_no=0
    # count_dic_viewwise={}
    # eval_subgroup = False
    # eval_mode = True
    # conf_mat_test=np.zeros((config_params['numclasses'],config_params['numclasses']))
    # views_standard=['LCC', 'LMLO', 'RCC', 'RMLO']

    # with torch.no_grad():
    #     for test_idx, test_batch, test_labels, views_names in dataloader_test:
    #         test_batch, test_labels = test_batch.to(config_params['device']), test_labels.to(config_params['device'])
    #         test_labels = test_labels.view(-1)

            
    #         output_batch_local, output_batch_global, output_batch_fusion, saliency_map, _, _, _, _ = model(test_batch) # compute model output, loss and total train loss over one epoch
    #         output_patch_test = None
            
    #         output_batch_local = output_batch_local.view(-1)
    #         output_batch_global = output_batch_global.view(-1)
    #         output_batch_fusion = output_batch_fusion.view(-1)
    #         test_labels = test_labels.float()
    #         test_pred = torch.ge(torch.sigmoid(output_batch_fusion), torch.tensor(0.5)).float()
    #         output_test = output_batch_fusion
            
            
    #         if batch_test_no==0:
    #             test_pred_all=test_pred
    #             test_labels_all=test_labels
    #             print(output_test.data.shape, flush=True)
    #             output_all_ten=torch.sigmoid(output_test.data)
    #         else:
    #             test_pred_all=torch.cat((test_pred_all,test_pred),dim=0)
    #             test_labels_all=torch.cat((test_labels_all,test_labels),dim=0)
    #             output_all_ten=torch.cat((output_all_ten,torch.sigmoid(output_test.data)),dim=0)
                
                
    #         correct, total_images, conf_mat_test, conf_mat_batch = evaluation.conf_mat_create(test_pred, test_labels, correct, total_images, conf_mat_test, config_params['classes'])
    #         batch_test_no+=1
    #         s=s+test_labels.shape[0]
    #         print('Test: Step [{}/{}], Loss: {:.4f}'.format(batch_test_no, batches_test, loss1), flush=True)
